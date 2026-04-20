"""
Heavy Data Augmentation Pipeline for Waste Classification

This module implements aggressive augmentation to handle:
- Dirty, broken, or overlapping objects
- Varying lighting conditions
- Real-world cluttered images

Augmentation includes all required transforms:
- Random horizontal flip
- Random brightness +/-0.15
- Random contrast +/-0.1
- Random crop with padding 10%

TASK 3 UPGRADES:
- A.RandomSunFlare(p=0.1)     — simulates harsh bin lighting
- A.RandomFog(p=0.1)          — simulates dirty/foggy camera lens
- A.Sharpen(p=0.3)            — improves texture edge learning
- A.ToGray(p=0.1)             — forces texture-over-color learning
- MixUp / CutMix alpha bumped to 0.5

TASK 5 UPGRADES:
- get_test_time_augmentation() expanded from 4 → 8 variants
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
    - Varying lighting (brightness, contrast, shadows, sun flare, fog)
    - Different camera angles (rotation, perspective)

    Required augmentations included:
    - Random horizontal flip
    - Random brightness +/-0.15
    - Random contrast +/-0.1
    - Random crop with padding 10%
    """
    pad_size = int(image_size * 0.1)
    return A.Compose([
        # Resize
        A.Resize(image_size, image_size),

        # Random translation via padding then cropping
        A.PadIfNeeded(
            min_height=image_size + pad_size * 2,
            min_width=image_size + pad_size * 2,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.RandomCrop(height=image_size, width=image_size, p=1.0),

        # Geometric transformations (camera angles, perspectives)
        A.OneOf([
            A.RandomRotate90(p=1),
            A.Rotate(limit=45, p=1),
        ], p=0.5),

        # Required: Random horizontal flip
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),

        A.OneOf([
            A.Perspective(scale=(0.05, 0.1), p=1),
            A.Affine(shear=(-15, 15), p=1),
        ], p=0.4),

        # Required: Random brightness +/-0.15 and contrast +/-0.1
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.1,
            p=0.7
        ),

        # Additional lighting variations
        A.OneOf([
            A.RandomGamma(gamma_limit=(80, 120), p=1),
            A.CLAHE(clip_limit=4.0, p=1),
        ], p=0.4),

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
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),  # using default noise types
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
                num_holes_range=(1, 8),
                hole_height_range=(int(image_size * 0.05), int(image_size * 0.1)),
                hole_width_range=(int(image_size * 0.05), int(image_size * 0.1)),
                fill_value=0,
                p=1
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.ElasticTransform(alpha=1, sigma=50, p=1),
        ], p=0.3),

        # Add random shadows
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=0.3
        ),

        # Simulate different image qualities
        A.OneOf([
            A.ImageCompression(quality_range=(60, 100), p=1),
            A.Downscale(scale_range=(0.5, 0.9), p=1),
        ], p=0.2),

        # ------------------------------------------------------------------
        # TASK 3: New augmentations for harsh real-world conditions
        # ------------------------------------------------------------------

        # Simulates harsh overhead lighting in bins / outdoor settings
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_range=(0, 1),
            num_flare_circles_range=(3, 8),
            src_radius=200,
            src_color=(255, 255, 255),
            p=0.1
        ),

        # Simulates foggy or dirty camera lens on bin cameras
        A.RandomFog(
            fog_coef_range=(0.1, 0.3),
            alpha_coef=0.08,
            p=0.1
        ),

        # Sharpens texture edges — helps distinguish glass vs. plastic vs. metal
        A.Sharpen(
            alpha=(0.2, 0.5),
            lightness=(0.5, 1.0),
            p=0.3
        ),

        # Forces the model to learn texture patterns over color cues
        A.ToGray(p=0.1),

        # ------------------------------------------------------------------

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
    Transform for inference/prediction (single-image, no TTA)
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
    Test-time augmentation (TTA) — Task 5 upgrade.

    Expanded from 4 → 8 variants for stronger ensemble effect:
      1. Original
      2. Horizontal flip
      3. Vertical flip              [NEW]
      4. Rotate 90°
      5. Rotate 180°                [NEW]
      6. Rotate 270°                [NEW]
      7. Horizontal flip + 90° rot  [NEW]
      8. Brightness +0.1

    Usage in inference:
        transforms = get_test_time_augmentation()
        logits_list = [model(t(image=img)['image'].unsqueeze(0)) for t in transforms]
        avg_probs = torch.stack([softmax(l) for l in logits_list]).mean(0)
        pred = avg_probs.argmax()
    """
    _norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return [
        # 1. Original
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(**_norm),
            ToTensorV2()
        ]),
        # 2. Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(**_norm),
            ToTensorV2()
        ]),
        # 3. Vertical flip [NEW]
        A.Compose([
            A.Resize(image_size, image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(**_norm),
            ToTensorV2()
        ]),
        # 4. Rotate 90°
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(90, 90), p=1.0),
            A.Normalize(**_norm),
            ToTensorV2()
        ]),
        # 5. Rotate 180° [NEW]
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(180, 180), p=1.0),
            A.Normalize(**_norm),
            ToTensorV2()
        ]),
        # 6. Rotate 270° [NEW]
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(270, 270), p=1.0),
            A.Normalize(**_norm),
            ToTensorV2()
        ]),
        # 7. Horizontal flip + 90° rotation [NEW]
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=(90, 90), p=1.0),
            A.Normalize(**_norm),
            ToTensorV2()
        ]),
        # 8. Brightness +0.1
        A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0),
            A.Normalize(**_norm),
            ToTensorV2()
        ]),
    ]


class MixUpAugmentation:
    """
    MixUp augmentation for improved generalization.
    Mixes two images and their labels.

    TASK 3: alpha bumped to 0.5 (was 0.4)
    """

    def __init__(self, alpha=0.5):
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
    CutMix augmentation for improved localization.
    Cuts and pastes patches between images.

    TASK 3: alpha bumped to 0.5 (was 1.0)
    """

    def __init__(self, alpha=0.5):
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

    # Test TTA — 8 variants
    tta_transforms = get_test_time_augmentation()
    print(f"\nTTA variants: {len(tta_transforms)} (expected 8)")
    for i, t in enumerate(tta_transforms):
        out = t(image=dummy_image)['image']
        print(f"  TTA variant {i+1}: shape={out.shape}")

    print("\nAugmentation pipeline test complete!")
