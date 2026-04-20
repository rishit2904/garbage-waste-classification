"""
CNN Model Architecture for Garbage Waste Classification

This module implements a robust CNN architecture using transfer learning
to handle real-world waste classification with:
- Pre-trained weights from ImageNet for better feature extraction
- Fine-tuned classification head for waste categories
- Increased model capacity (25% more hidden units)
- Regularized with dropout >= 0.3

TASK 1 UPGRADE: EfficientNet-B2 backbone with timm (fallback: torchvision)
- Selective fine-tuning: first 60% of layers frozen
- Custom head: Dropout(0.4) -> Linear(1408, 512) -> GELU -> Dropout(0.25) -> Linear(512, 6)
- Two-phase training helpers built in
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights

# Try to import timm for EfficientNet-B2; fall back gracefully
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Waste categories
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# EfficientNet-B2 feature dimension (fixed by architecture)
EFFICIENTNET_B2_FEATURES = 1408


# ---------------------------------------------------------------------------
# Task 1: EfficientNet-B2 (primary upgrade)
# ---------------------------------------------------------------------------

class GarbageClassifierEfficientNet(nn.Module):
    """
    Transfer learning classifier using EfficientNet-B2 backbone.

    Strategy:
    - Freeze first 60% of parameter groups (low-level features)
    - Unfreeze upper 40% for selective fine-tuning
    - Custom head with GELU and two dropout stages
    - Two-phase training helpers for head-only vs. full fine-tuning

    Head architecture (Task 1):
        Dropout(0.4) -> Linear(1408, 512) -> GELU -> Dropout(0.25) -> Linear(512, 6)
    """

    CLASSES = CLASSES

    def __init__(self, num_classes=6, pretrained=True):
        super(GarbageClassifierEfficientNet, self).__init__()

        # -----------------------------------------------------------------
        # Build backbone: timm preferred, torchvision as fallback
        # -----------------------------------------------------------------
        if TIMM_AVAILABLE:
            # num_classes=0 strips the timm classifier so we can attach ours
            self.backbone = timm.create_model(
                'efficientnet_b2',
                pretrained=pretrained,
                num_classes=0,          # returns feature vector only
                global_pool='avg'       # global average pooling built-in
            )
            in_features = self.backbone.num_features  # 1408 for B2
            self._use_timm = True
        else:
            try:
                from torchvision.models import EfficientNet_B2_Weights
                weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            except ImportError:
                weights = 'DEFAULT' if pretrained else None
            self.backbone = models.efficientnet_b2(weights=weights)
            in_features = self.backbone.classifier[1].in_features  # 1408
            # Remove torchvision's classifier so backbone outputs features
            self.backbone.classifier = nn.Identity()
            self._use_timm = False

        # -----------------------------------------------------------------
        # Custom classification head (Task 1 spec)
        # -----------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes)
        )

        # -----------------------------------------------------------------
        # Selective freeze: first 60% of named parameter blocks
        # -----------------------------------------------------------------
        self._freeze_first_60_percent()

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _freeze_first_60_percent(self):
        """Freeze the first 60% of all backbone parameter tensors."""
        all_params = list(self.backbone.named_parameters())
        freeze_count = int(len(all_params) * 0.60)
        for i, (name, param) in enumerate(all_params):
            param.requires_grad = (i >= freeze_count)

    def unfreeze_upper_layers(self):
        """
        Phase 2: unfreeze upper 40% of backbone (already done at init,
        but called explicitly when entering Phase 2 to be explicit).
        Also ensures classifier head grads are on.
        """
        all_params = list(self.backbone.named_parameters())
        freeze_count = int(len(all_params) * 0.60)
        for i, (name, param) in enumerate(all_params):
            param.requires_grad = (i >= freeze_count)
        for param in self.classifier.parameters():
            param.requires_grad = True

    def freeze_backbone_fully(self):
        """Phase 1: freeze entire backbone, only train classifier head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Phase parameter helpers (for optimizer construction)
    # ------------------------------------------------------------------

    def get_phase1_params(self):
        """Return only classifier head parameters (Phase 1 training)."""
        return self.classifier.parameters()

    def get_phase2_params(self):
        """Return all trainable parameters (Phase 2 fine-tuning)."""
        return filter(lambda p: p.requires_grad, self.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        features = self.backbone(x)
        # timm with global_pool='avg' and num_classes=0 already flattens
        # torchvision with Identity classifier also gives flat vector
        if features.dim() > 2:
            features = features.flatten(1)
        return self.classifier(features)

    def predict(self, x):
        """Get class predictions with probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_model_summary(self):
        """Print trainable/frozen param counts and improvement rationale."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        backbone_str = "timm (EfficientNet-B2)" if self._use_timm else "torchvision (EfficientNet-B2)"

        print("\n" + "=" * 60)
        print("  Model Summary — GarbageClassifierEfficientNet")
        print("=" * 60)
        print(f"  Backbone        : {backbone_str}")
        print(f"  Total params    : {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Frozen params   : {frozen:,}")
        print(f"  Freeze ratio    : {frozen/total*100:.1f}% frozen")
        print("-" * 60)
        print("  Estimated accuracy gains over ResNet18 baseline (94.7%):")
        print("    EfficientNet-B2 backbone     : +1.0–1.5%")
        print("    Class weights + WRS sampler  : +0.3–0.5%")
        print("    Label smoothing (0.1)        : +0.2–0.3%")
        print("    Enhanced augmentation        : +0.2–0.4%")
        print("    8-variant TTA at inference   : +0.3–0.5%")
        print("    Two-phase training           : +0.2–0.4%")
        print("    ─────────────────────────────────────────")
        print("    Total estimated              : +2.2–3.6%")
        print("    Target accuracy              : ~96.9–98.3%")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Existing models (unchanged — backward compatible)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_model(model_type='efficientnet', num_classes=6, pretrained=True, pretrained_path=None):
    """
    Factory function to get the appropriate model

    Args:
        model_type: 'efficientnet' (NEW default, EfficientNet-B2),
                    'transfer' (ResNet18), 'mobilenet', or 'cnn' (custom scratch)
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        pretrained_path: Path to custom pretrained weights

    Returns:
        Model instance
    """
    if model_type == 'efficientnet':
        model = GarbageClassifierEfficientNet(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'transfer' or model_type == 'resnet':
        model = GarbageClassifierTransfer(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'mobilenet':
        model = GarbageClassifierMobileNet(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'cnn':
        model = GarbageClassifierCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Valid options: 'efficientnet', 'transfer', 'mobilenet', 'cnn'")

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)

    print("Testing GarbageClassifierEfficientNet (B2)...")
    model_eff = GarbageClassifierEfficientNet()
    output = model_eff(x)
    print(f"  EfficientNet-B2 output shape: {output.shape}")
    model_eff.print_model_summary()

    print("Testing GarbageClassifierTransfer (ResNet18)...")
    model_transfer = GarbageClassifierTransfer(pretrained=False)
    output = model_transfer(x)
    print(f"  Transfer output shape: {output.shape}")

    print("\nTesting GarbageClassifierMobileNet...")
    model_mobile = GarbageClassifierMobileNet(pretrained=False)
    output = model_mobile(x)
    print(f"  MobileNet output shape: {output.shape}")

    print("\nTesting GarbageClassifierCNN...")
    model_cnn = GarbageClassifierCNN()
    output = model_cnn(x)
    print(f"  CNN output shape: {output.shape}")

    # Print model parameters
    total_params_eff      = sum(p.numel() for p in model_eff.parameters())
    total_params_transfer = sum(p.numel() for p in model_transfer.parameters())
    total_params_mobile   = sum(p.numel() for p in model_mobile.parameters())
    total_params_cnn      = sum(p.numel() for p in model_cnn.parameters())

    print(f"\nEfficientNet-B2 total parameters   : {total_params_eff:,}")
    print(f"Transfer (ResNet18) total parameters: {total_params_transfer:,}")
    print(f"MobileNet total parameters          : {total_params_mobile:,}")
    print(f"CNN total parameters                : {total_params_cnn:,}")
