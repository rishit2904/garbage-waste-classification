"""
CNN Model Architecture for Garbage Waste Classification

This module implements a robust CNN architecture designed to handle:
- Dirty, broken, or overlapping waste objects
- Varying lighting conditions
- Visual confusion between similar materials (plastic vs glass)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GarbageClassifierCNN(nn.Module):
    """
    Custom CNN architecture for waste classification with:
    - Deep convolutional layers with batch normalization
    - Dropout for regularization
    - Global average pooling for spatial invariance
    """
    
    # Waste categories
    CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(GarbageClassifierCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def predict(self, x):
        """Get class predictions with probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GarbageClassifierResNet(nn.Module):
    """
    ResNet-style architecture for improved feature learning
    Better for handling complex, real-world waste images
    """
    
    CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    def __init__(self, num_classes=6):
        super(GarbageClassifierResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def predict(self, x):
        """Get class predictions with probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


def get_model(model_type='cnn', num_classes=6, pretrained_path=None):
    """
    Factory function to get the appropriate model
    
    Args:
        model_type: 'cnn' or 'resnet'
        num_classes: Number of output classes
        pretrained_path: Path to pretrained weights
    
    Returns:
        Model instance
    """
    if model_type == 'cnn':
        model = GarbageClassifierCNN(num_classes=num_classes)
    elif model_type == 'resnet':
        model = GarbageClassifierResNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    
    return model


if __name__ == "__main__":
    # Test the models
    print("Testing GarbageClassifierCNN...")
    model_cnn = GarbageClassifierCNN()
    x = torch.randn(4, 3, 224, 224)
    output = model_cnn(x)
    print(f"CNN Output shape: {output.shape}")
    
    print("\nTesting GarbageClassifierResNet...")
    model_resnet = GarbageClassifierResNet()
    output = model_resnet(x)
    print(f"ResNet Output shape: {output.shape}")
    
    # Print model parameters
    total_params_cnn = sum(p.numel() for p in model_cnn.parameters())
    total_params_resnet = sum(p.numel() for p in model_resnet.parameters())
    print(f"\nCNN Total parameters: {total_params_cnn:,}")
    print(f"ResNet Total parameters: {total_params_resnet:,}")
