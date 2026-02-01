"""
Heavy Data Augmentation Pipeline for Waste Classification

This module implements aggressive augmentation to handle:
- Dirty, broken, or overlapping objects
- Varying lighting conditions
- Real-world cluttered images
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_training_augmentation(image_size=224):
    """
    Heavy augmentation pipeline for training
    
    Simulates real-world conditions:
    - Dirty objects (noise, blur)
    - Broken/overlapping objects (cutout, grid distortion)
    - Varying lighting (brightness, contrast, shadows)
    - Different camera angles (rotation, perspective)
    """
    return A.Compose([
        # Resize
        A.Resize(image_size, image_size),
        
        # Geometric transformations (camera angles, perspectives)
        A.OneOf([
            A.RandomRotate90(p=1),
            A.Rotate(limit=45, p=1),
        ], p=0.5),
        
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        
        A.OneOf([
            A.Perspective(scale=(0.05, 0.1), p=1),
            A.Affine(shear=(-15, 15), p=1),
        ], p=0.4),
        
        # Simulate varying lighting conditions
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
            A.CLAHE(clip_limit=4.0, p=1),
        ], p=0.7),
        
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
        ], p=0.5),
        
        # Simulate dirty/weathered objects
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.ISONoise(intensity=(0.1, 0.5), p=1),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
        ], p=0.4),
        
        # Simulate blur from motion or focus issues
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1),
            A.GaussianBlur(blur_limit=7, p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.3),
        
        # Simulate overlapping/partial occlusion
        A.OneOf([
            A.CoarseDropout(
                max_holes=8,
                max_height=int(image_size * 0.1),
                max_width=int(image_size * 0.1),
                fill_value=0,
                p=1
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.ElasticTransform(alpha=1, sigma=50, p=1),
        ], p=0.3),
        
        # Add random shadows
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=3,
            shadow_dimension=5,
            p=0.3
        ),
        
        # Simulate different image qualities
        A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=1),
            A.Downscale(scale_min=0.5, scale_max=0.9, p=1),
        ], p=0.2),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_validation_augmentation(image_size=224):
    """
    Minimal augmentation for validation/testing
    Only resize and normalize
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_inference_transform(image_size=224):
    """
    Transform for inference/prediction
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_test_time_augmentation(image_size=224):
    """
    Test-time augmentation for improved prediction accuracy
    Returns multiple augmented versions of the same image
    """
    return [
        # Original
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Rotate 90
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(90, 90), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Brightness adjustment
        A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]


class MixUpAugmentation:
    """
    MixUp augmentation for improved generalization
    Mixes two images and their labels
    """
    
    def __init__(self, alpha=0.4):
        self.alpha = alpha
    
    def __call__(self, img1, img2, label1, label2):
        """
        Mix two images and labels
        
        Args:
            img1, img2: Input images (tensors)
            label1, label2: One-hot encoded labels
        
        Returns:
            Mixed image and mixed label
        """
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_img, mixed_label


class CutMixAugmentation:
    """
    CutMix augmentation for improved localization
    Cuts and pastes patches between images
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, img1, img2, label1, label2):
        """
        Apply CutMix to two images
        """
        lam = np.random.beta(self.alpha, self.alpha)
        
        _, H, W = img1.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center position
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        mixed_img = img1.clone()
        mixed_img[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda for the actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_img, mixed_label


def visualize_augmentations(image_path, save_path=None, num_samples=6):
    """
    Visualize augmentation effects on a sample image
    
    Args:
        image_path: Path to input image
        save_path: Optional path to save visualization
        num_samples: Number of augmented samples to show
    """
    import matplotlib.pyplot as plt
    
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get augmentation pipeline
    transform = get_training_augmentation()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show augmented versions
    for i in range(1, num_samples):
        augmented = transform(image=image)['image']
        # Denormalize for visualization
        augmented = augmented.permute(1, 2, 0).numpy()
        augmented = augmented * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        augmented = np.clip(augmented, 0, 1)
        
        axes[i].imshow(augmented)
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test augmentation pipeline
    print("Testing augmentation pipeline...")
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Test training augmentation
    train_transform = get_training_augmentation()
    augmented = train_transform(image=dummy_image)
    print(f"Training augmentation output shape: {augmented['image'].shape}")
    
    # Test validation augmentation
    val_transform = get_validation_augmentation()
    augmented = val_transform(image=dummy_image)
    print(f"Validation augmentation output shape: {augmented['image'].shape}")
    
    print("\nAugmentation pipeline test complete!")
