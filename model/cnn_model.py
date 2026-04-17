"""
CNN Model Architecture for Garbage Waste Classification

This module implements a robust CNN architecture using transfer learning
to handle real-world waste classification with:
- Pre-trained weights from ImageNet for better feature extraction
- Fine-tuned classification head for waste categories
- Increased model capacity (25% more hidden units)
- Regularized with dropout >= 0.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights


# Waste categories
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


class GarbageClassifierTransfer(nn.Module):
    """
    Transfer learning based classifier using pre-trained ResNet18.
    Much better for real-world classification even without fine-tuning.
    Hidden layer increased from 256 to 320 (25% increase, nearest 64-multiple).
    """

    CLASSES = CLASSES

    def __init__(self, num_classes=6, pretrained=True):
        super(GarbageClassifierTransfer, self).__init__()

        # Load pre-trained ResNet18
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.backbone = models.resnet18(weights=None)

        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace the final fully connected layer
        # Hidden layer: 256 -> 320 (25% increase, nearest 64-multiple)
        # Dropout: 0.3 minimum
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def predict(self, x):
        """Get class predictions with probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


class GarbageClassifierMobileNet(nn.Module):
    """
    Lightweight classifier using MobileNetV2 for faster inference.
    Good for mobile/edge deployment.
    """

    CLASSES = CLASSES

    def __init__(self, num_classes=6, pretrained=True):
        super(GarbageClassifierMobileNet, self).__init__()

        # Load pre-trained MobileNetV2
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            self.backbone = models.mobilenet_v2(weights=None)

        # Freeze feature extractor
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def predict(self, x):
        """Get class predictions with probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


class GarbageClassifierCNN(nn.Module):
    """
    Custom CNN architecture for waste classification with:
    - Deep convolutional layers with batch normalization
    - Dropout for regularization (minimum 0.3)
    - Global average pooling for spatial invariance
    - Increased capacity: FC layers 512->640, 256->320
    """

    CLASSES = CLASSES

    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(GarbageClassifierCNN, self).__init__()

        # Ensure dropout is at least 0.3
        dropout_rate = max(dropout_rate, 0.3)

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
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
            nn.Dropout2d(0.3)
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
            nn.Dropout2d(0.3)
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
            nn.Dropout2d(0.3)
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers - increased capacity by 25%
        # 512 -> 640 (nearest 64-multiple of 512*1.25=640)
        # 256 -> 320 (nearest 64-multiple of 256*1.25=320)
        self.classifier = nn.Sequential(
            nn.Linear(256, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(640, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(320, num_classes)
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


def get_model(model_type='transfer', num_classes=6, pretrained=True, pretrained_path=None):
    """
    Factory function to get the appropriate model

    Args:
        model_type: 'transfer' (ResNet18), 'mobilenet', or 'cnn' (custom)
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        pretrained_path: Path to custom pretrained weights

    Returns:
        Model instance
    """
    if model_type == 'transfer':
        model = GarbageClassifierTransfer(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'mobilenet':
        model = GarbageClassifierMobileNet(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'cnn':
        model = GarbageClassifierCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    return model


if __name__ == "__main__":
    # Test the models
    print("Testing GarbageClassifierTransfer (ResNet18)...")
    model_transfer = GarbageClassifierTransfer()
    x = torch.randn(4, 3, 224, 224)
    output = model_transfer(x)
    print(f"Transfer Output shape: {output.shape}")

    print("\nTesting GarbageClassifierMobileNet...")
    model_mobile = GarbageClassifierMobileNet()
    output = model_mobile(x)
    print(f"MobileNet Output shape: {output.shape}")

    print("\nTesting GarbageClassifierCNN...")
    model_cnn = GarbageClassifierCNN()
    output = model_cnn(x)
    print(f"CNN Output shape: {output.shape}")

    # Print model parameters
    total_params_transfer = sum(p.numel() for p in model_transfer.parameters())
    total_params_mobile = sum(p.numel() for p in model_mobile.parameters())
    total_params_cnn = sum(p.numel() for p in model_cnn.parameters())
    print(f"\nTransfer (ResNet18) Total parameters: {total_params_transfer:,}")
    print(f"MobileNet Total parameters: {total_params_mobile:,}")
    print(f"CNN Total parameters: {total_params_cnn:,}")
