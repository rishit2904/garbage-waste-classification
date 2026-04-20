"""
Training Pipeline for Garbage Waste Classification

Features:
- Heavy data augmentation
- Early stopping with increased patience
- Learning rate scheduling
- Model checkpointing (best val loss)
- Mixed precision training
- Data deduplication and cleaning
- Per-epoch training log (training_log.json)
- Precision, recall, F1 evaluation
- Training summary printout

TASK 2 UPGRADES:
- Class-weighted CrossEntropyLoss (inverse-frequency weights from training set)
- WeightedRandomSampler for minority class oversampling

TASK 4 UPGRADES:
- Two-phase training for EfficientNet:
    Phase 1 (epochs 1–10): classifier head only, LR=1e-3
    Phase 2 (epochs 11–50): unfreeze upper layers, LR=1e-4 + CosineAnnealingWarmRestarts
- Label smoothing: CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)

TASK 6 UPGRADES:
- LR range test (manual or torch-lr-finder), triggered by --lr-finder flag

BACKWARD COMPATIBILITY:
- cnn / transfer / mobilenet model types use single-phase training (original behavior)
- All original CLI args preserved; only --model-type choices and default updated
"""

import os
import sys
import argparse
import time
import hashlib
import unicodedata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import Counter
import json

from cnn_model import get_model, GarbageClassifierCNN
from augmentation import get_training_augmentation, get_validation_augmentation


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GarbageDataset(Dataset):
    """
    Custom dataset for garbage classification

    Expected folder structure:
    data/
        cardboard/
            img1.jpg
            img2.jpg
        glass/
            ...
        metal/
            ...
        paper/
            ...
        plastic/
            ...
        trash/
            ...
    """

    CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Get label
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def compute_file_hash(filepath):
    """Compute MD5 hash of a file for deduplication."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def deduplicate_dataset(image_paths, labels):
    """Remove exact duplicate files from the dataset based on file hash."""
    seen_hashes = set()
    unique_paths = []
    unique_labels = []
    dup_count = 0

    for path, label in zip(image_paths, labels):
        try:
            file_hash = compute_file_hash(path)
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_paths.append(path)
                unique_labels.append(label)
            else:
                dup_count += 1
        except Exception:
            # If we can't read the file, skip it
            continue

    if dup_count > 0:
        print(f"  Removed {dup_count} duplicate files")

    return unique_paths, unique_labels


def clean_path_string(s):
    """Strip whitespace and normalize unicode for a string."""
    if isinstance(s, str):
        s = s.strip()
        s = unicodedata.normalize('NFC', s)
    return s


def compute_class_weights(labels, num_classes=6, device='cpu'):
    """
    Task 2: Compute inverse-frequency class weights for CrossEntropyLoss.

    Formula: weight[c] = total_samples / (num_classes * count[c])
    This upweights minority classes (e.g., trash=137) relative to
    majority classes (e.g., paper=594).

    Args:
        labels: list of integer class labels from training set
        num_classes: number of classes (default 6)
        device: torch device string

    Returns:
        FloatTensor of shape [num_classes] on the given device
    """
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        count = counts.get(c, 1)  # avoid division by zero
        weights.append(total / (num_classes * count))
    weight_tensor = torch.FloatTensor(weights).to(device)
    return weight_tensor


def build_weighted_sampler(labels):
    """
    Task 2: Build a WeightedRandomSampler so that each sample's
    probability is proportional to 1/class_frequency.

    This oversamples minority classes (trash) and undersamples
    majority classes (paper) to balance the effective training distribution.

    Args:
        labels: list of integer class labels from training set

    Returns:
        WeightedRandomSampler instance
    """
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(counts)

    # Weight for each class: inverse frequency
    class_weights = {c: total / (num_classes * cnt) for c, cnt in counts.items()}

    # Assign per-sample weight
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler


def load_dataset(data_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Load and split dataset into train, validation, and test sets

    Includes:
    - Data cleaning (whitespace strip, unicode NFC normalization)
    - Deduplication (exact file hash)
    - Stratified train/val/test split (70/15/15 — NOT changed per constraints)

    Args:
        data_dir: Path to data directory
        test_size: Proportion for test set
        val_size: Proportion for validation set (from training data)
        random_state: Random seed

    Returns:
        Train, validation, and test datasets
    """
    image_paths = []
    labels = []

    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue

        for img_name in os.listdir(class_dir):
            # Clean filename
            img_name_clean = clean_path_string(img_name)
            if img_name_clean.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                img_path = os.path.join(class_dir, img_name)
                # Skip empty/null files
                if os.path.getsize(img_path) == 0:
                    continue
                image_paths.append(img_path)
                labels.append(class_idx)

    if len(image_paths) == 0:
        print("No images found! Creating synthetic data for demo...")
        return None, None, None

    # Deduplication step
    print("\nData Cleaning:")
    print(f"  Total files found: {len(image_paths)}")
    image_paths, labels = deduplicate_dataset(image_paths, labels)
    print(f"  Files after deduplication: {len(image_paths)}")

    # Split data: first split off test set (70/15/15 stratified — unchanged)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        image_paths, labels, test_size=test_size,
        random_state=random_state, stratify=labels
    )

    # Then split train into train and validation (stratified)
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction,
        random_state=random_state, stratify=y_trainval
    )

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"\nClass distribution (training):")
    for cls, count in sorted(Counter(y_train).items()):
        print(f"  {classes[cls]}: {count}")

    # Create datasets
    train_transform = get_training_augmentation()
    val_transform = get_validation_augmentation()

    train_dataset = GarbageDataset(X_train, y_train, transform=train_transform)
    val_dataset = GarbageDataset(X_val, y_val, transform=val_transform)
    test_dataset = GarbageDataset(X_test, y_test, transform=val_transform)

    # Attach raw labels for sampler / class weight computation
    train_dataset.raw_labels = y_train

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict().copy()
            self.status = f"Improvement detected, reset counter"
        else:
            self.counter += 1
            self.status = f"No improvement, counter: {self.counter}/{self.patience}"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


# ---------------------------------------------------------------------------
# Train / validate epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate_metrics(model, dataloader, device):
    """Evaluate model with precision, recall, F1 (micro averaging for multiclass)."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='micro')
    recall = recall_score(all_labels, all_preds, average='micro')
    f1 = f1_score(all_labels, all_preds, average='micro')

    return acc, precision, recall, f1


def append_training_log(log_path, epoch_data):
    """Append epoch data to training_log.json in append mode."""
    log = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log = json.load(f)
        except (json.JSONDecodeError, IOError):
            log = []

    log.append(epoch_data)

    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


# ---------------------------------------------------------------------------
# Task 6: LR Finder
# ---------------------------------------------------------------------------

def run_lr_finder(model, train_loader, device, save_dir, num_steps=100,
                  start_lr=1e-7, end_lr=1.0):
    """
    Task 6: LR range test over `num_steps` mini-batches.

    Uses torch-lr-finder if available; falls back to a manual
    exponential sweep that logs loss vs. LR and suggests
    the LR at the steepest loss decline.

    Args:
        model: PyTorch model (should be freshly initialized)
        train_loader: DataLoader for training set
        device: torch device
        save_dir: directory to save lr_finder_plot.png
        num_steps: number of steps to sweep over (default 100)
        start_lr: starting LR (default 1e-7)
        end_lr: ending LR (default 1.0)

    Returns:
        Suggested LR (float)
    """
    print("\n" + "=" * 50)
    print("  Running LR Range Test...")
    print("=" * 50)

    # Try torch-lr-finder first
    try:
        from torch_lr_finder import LRFinder
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=start_lr, weight_decay=0.01
        )
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, start_lr=start_lr, end_lr=end_lr,
                             num_iter=num_steps, smooth_f=0.05)

        # Find steepest decline
        lrs = lr_finder.history['lr']
        losses = lr_finder.history['loss']
        min_grad_idx = (np.gradient(np.array(losses))).argmin()
        suggested_lr = lrs[min_grad_idx]

        # Save plot
        fig, ax = plt.subplots()
        ax.plot(lrs, losses)
        ax.set_xscale('log')
        ax.axvline(x=suggested_lr, color='r', linestyle='--',
                   label=f'Suggested LR = {suggested_lr:.2e}')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Loss')
        ax.set_title('LR Range Test')
        ax.legend()
        plot_path = os.path.join(save_dir, 'lr_finder_plot.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # Reset model/optimizer state
        lr_finder.reset()

        print(f"  torch-lr-finder used. Suggested LR: {suggested_lr:.2e}")
        print(f"  Plot saved to: {plot_path}")
        return suggested_lr

    except ImportError:
        pass  # Fall back to manual implementation

    # Manual LR range test
    print("  (torch-lr-finder not installed — using manual sweep)")

    criterion = nn.CrossEntropyLoss()
    # Save initial model state to restore after range test
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=start_lr, weight_decay=0.01
    )

    lrs, losses = [], []
    lr = start_lr
    lr_mult = (end_lr / start_lr) ** (1.0 / num_steps)
    best_loss = float('inf')
    smoothed_loss = None
    beta = 0.98  # smoothing factor

    model.train()
    data_iter = iter(train_loader)
    for step in range(num_steps):
        # Fetch next batch (cycle through loader if needed)
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        images, labels = images.to(device), labels.to(device)

        # Set LR
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Exponential smoothing
        raw_loss = loss.item()
        smoothed_loss = beta * (smoothed_loss or raw_loss) + (1 - beta) * raw_loss
        debiased_loss = smoothed_loss / (1 - beta ** (step + 1))

        lrs.append(lr)
        losses.append(debiased_loss)

        # Stop if loss diverges
        if debiased_loss < best_loss:
            best_loss = debiased_loss
        if debiased_loss > 4 * best_loss:
            print(f"  Loss diverged at step {step}. Stopping early.")
            break

        lr *= lr_mult

    # Find steepest decline (best LR = just before the minimum)
    if len(losses) > 5:
        gradients = np.gradient(np.array(losses))
        min_grad_idx = gradients.argmin()
        suggested_lr = lrs[min_grad_idx]
    else:
        suggested_lr = 1e-3  # safe default

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(lrs[:len(losses)], losses)
    ax.set_xscale('log')
    ax.axvline(x=suggested_lr, color='r', linestyle='--',
               label=f'Suggested LR = {suggested_lr:.2e}')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Smoothed Loss')
    ax.set_title('LR Range Test (manual)')
    ax.legend()
    plot_path = os.path.join(save_dir, 'lr_finder_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Restore model weights (range test should not affect training)
    model.load_state_dict(initial_state)

    print(f"  Suggested LR from range test: {suggested_lr:.2e}")
    print(f"  Plot saved to: {plot_path}")
    return suggested_lr


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config):
    """
    Main training function.

    For 'efficientnet' model type, uses two-phase training (Tasks 4).
    For all other model types, uses single-phase training (backward compatible).

    Args:
        config: Training configuration dictionary
    """
    start_time = time.time()

    # Setup device (CUDA > MPS > CPU)
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"\nUsing device: {device}")

    # Load datasets (70/15/15 split — unchanged per constraints)
    train_dataset, val_dataset, test_dataset = load_dataset(
        config['data_dir'],
        test_size=config.get('test_size', 0.15),
        val_size=config.get('val_size', 0.15)
    )

    if train_dataset is None:
        print("\nNo dataset found. Please add images to the data folder.")
        print("Expected structure:")
        print("  data/")
        print("    cardboard/")
        print("    glass/")
        print("    metal/")
        print("    paper/")
        print("    plastic/")
        print("    trash/")

        # Create a dummy model and save it for demo
        print("\nCreating untrained model for demo purposes...")
        model = get_model(config.get('model_type', 'efficientnet'))
        os.makedirs(config['save_dir'], exist_ok=True)
        save_path = os.path.join(config['save_dir'], 'garbage_classifier.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Untrained model saved to: {save_path}")
        return

    # ------------------------------------------------------------------
    # Task 2: Class weights + WeightedRandomSampler
    # ------------------------------------------------------------------
    raw_labels = getattr(train_dataset, 'raw_labels', train_dataset.labels)

    class_weights = compute_class_weights(raw_labels, num_classes=6, device=device)
    print(f"\nClass weights (inverse-frequency):")
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    for i, (cls, w) in enumerate(zip(classes, class_weights.cpu().tolist())):
        print(f"  {cls}: {w:.4f}")

    sampler = build_weighted_sampler(raw_labels)

    # Task 4: Label smoothing + class weights in loss
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    # ------------------------------------------------------------------
    # Create data loaders
    # ------------------------------------------------------------------
    # Use WeightedRandomSampler (Task 2) — sampler replaces shuffle=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,          # Task 2: oversamples minority classes
        num_workers=config.get('num_workers', 0),
        pin_memory=(device.type != 'cpu'),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=(device.type != 'cpu')
    )

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------
    model_type = config.get('model_type', 'efficientnet')
    model = get_model(model_type)
    model = model.to(device)

    # Print EfficientNet summary if applicable
    if model_type == 'efficientnet' and hasattr(model, 'print_model_summary'):
        model.print_model_summary()

    # ------------------------------------------------------------------
    # Task 6: Optional LR finder before training
    # ------------------------------------------------------------------
    os.makedirs(config['save_dir'], exist_ok=True)
    if config.get('run_lr_finder', False):
        suggested_lr = run_lr_finder(
            model, train_loader, device, config['save_dir'], num_steps=100
        )
        print(f"\n  Using suggested LR from range test: {suggested_lr:.2e}")
        config['learning_rate'] = suggested_lr
    
    # ------------------------------------------------------------------
    # Task 4: Two-phase training (EfficientNet only)
    # Single-phase training for cnn / transfer / mobilenet (backward compat)
    # ------------------------------------------------------------------
    is_efficientnet = (model_type == 'efficientnet')

    if is_efficientnet:
        _train_two_phase(model, train_loader, val_loader, criterion, device, config)
    else:
        _train_single_phase(model, train_loader, val_loader, criterion, device, config)

    # ------------------------------------------------------------------
    # Post-training: evaluate on test set
    # ------------------------------------------------------------------
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 0)
        )
        test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
        test_acc_full, test_precision, test_recall, test_f1 = evaluate_metrics(
            model, test_loader, device
        )

        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc_full:.2f}%")
        print(f"  Precision (micro): {test_precision:.4f}")
        print(f"  Recall (micro): {test_recall:.4f}")
        print(f"  F1 Score (micro): {test_f1:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")


# ---------------------------------------------------------------------------
# Task 4: Two-phase training loop (EfficientNet)
# ---------------------------------------------------------------------------

def _train_two_phase(model, train_loader, val_loader, criterion, device, config):
    """
    Two-phase training strategy for EfficientNet-B2 (Task 4):

    Phase 1 (epochs 1–10):
        - Freeze backbone entirely, train only classifier head
        - Optimizer: AdamW, LR=1e-3
        - No scheduler, no early stopping

    Phase 2 (epochs 11–end):
        - Unfreeze upper backbone layers
        - Optimizer: AdamW, LR=1e-4
        - Scheduler: CosineAnnealingWarmRestarts(T_0=20, T_mult=2)
        - Early stopping active (patience from config)
    """
    os.makedirs(config['save_dir'], exist_ok=True)
    checkpoint_dir = os.path.join(
        os.path.dirname(config['save_dir']), 'checkpoints', 'best_model'
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(config['save_dir'], 'training_log.json')

    scaler = GradScaler() if device.type == 'cuda' else None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    total_epochs = config['epochs']
    phase1_epochs = min(10, total_epochs)
    actual_epochs = 0

    print(f"\n{'='*55}")
    print(f"  TWO-PHASE TRAINING  |  Total epochs: {total_epochs}")
    print(f"  Phase 1 (head only) : epochs 1–{phase1_epochs}, LR=1e-3")
    print(f"  Phase 2 (fine-tune) : epochs {phase1_epochs+1}–{total_epochs}, LR=1e-4")
    print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # PHASE 1: Classifier head only
    # ------------------------------------------------------------------
    model.freeze_backbone_fully()
    optimizer_p1 = optim.AdamW(
        model.get_phase1_params(),
        lr=1e-3,
        weight_decay=config.get('weight_decay', 0.01)
    )

    print(f"--- Phase 1: Training classifier head only ---")
    for epoch in range(phase1_epochs):
        actual_epochs += 1
        print(f"\nEpoch {actual_epochs}/{total_epochs}  [Phase 1]")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer_p1, device, scaler
        )
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        current_lr = optimizer_p1.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        append_training_log(log_path, {
            'epoch': actual_epochs, 'phase': 1,
            'train_loss': round(train_loss, 6), 'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 6), 'val_acc': round(val_acc, 4),
            'learning_rate': round(current_lr, 8)
        })

        # Checkpoint on best val loss during Phase 1 as well
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            _save_checkpoint(model, optimizer_p1, epoch, val_acc, val_loss, checkpoint_dir)

    # ------------------------------------------------------------------
    # PHASE 2: Fine-tune upper backbone + head
    # ------------------------------------------------------------------
    if total_epochs > phase1_epochs:
        model.unfreeze_upper_layers()

        optimizer_p2 = optim.AdamW(
            model.get_phase2_params(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_p2, T_0=20, T_mult=2
        )
        early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            restore_best_weights=True
        )

        print(f"\n--- Phase 2: Fine-tuning upper layers ---")
        for epoch in range(phase1_epochs, total_epochs):
            actual_epochs += 1
            print(f"\nEpoch {actual_epochs}/{total_epochs}  [Phase 2]")

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer_p2, device, scaler
            )
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            scheduler.step()
            current_lr = optimizer_p2.param_groups[0]['lr']

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            append_training_log(log_path, {
                'epoch': actual_epochs, 'phase': 2,
                'train_loss': round(train_loss, 6), 'train_acc': round(train_acc, 4),
                'val_loss': round(val_loss, 6), 'val_acc': round(val_acc, 4),
                'learning_rate': round(current_lr, 8)
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                _save_checkpoint(model, optimizer_p2, epoch, val_acc, val_loss, checkpoint_dir)
                print(f"  -> Best checkpoint saved (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if early_stopping(val_loss, model):
                print(f"\nEarly stopping triggered at epoch {actual_epochs}")
                break
            print(f"  {early_stopping.status}")

    # Save final model + history
    _finalize_training(model, history, config, best_val_loss, best_val_acc, actual_epochs)


# ---------------------------------------------------------------------------
# Backward-compatible single-phase training (cnn / transfer / mobilenet)
# ---------------------------------------------------------------------------

def _train_single_phase(model, train_loader, val_loader, criterion, device, config):
    """
    Original single-phase training loop.
    Preserved for backward compatibility with --model-type cnn/transfer/mobilenet.
    """
    os.makedirs(config['save_dir'], exist_ok=True)
    checkpoint_dir = os.path.join(
        os.path.dirname(config['save_dir']), 'checkpoints', 'best_model'
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(config['save_dir'], 'training_log.json')

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    early_stopping = EarlyStopping(
        patience=config.get('patience', 15),
        restore_best_weights=True
    )
    scaler = GradScaler() if device.type == 'cuda' else None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    actual_epochs = 0

    print(f"\nStarting single-phase training for {config['epochs']} epochs...")

    for epoch in range(config['epochs']):
        actual_epochs = epoch + 1
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        append_training_log(log_path, {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6), 'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 6), 'val_acc': round(val_acc, 4),
            'learning_rate': round(current_lr, 8)
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            _save_checkpoint(model, optimizer, epoch, val_acc, val_loss, checkpoint_dir)
            print(f"  -> Best checkpoint saved (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
        print(f"  {early_stopping.status}")

    _finalize_training(model, history, config, best_val_loss, best_val_acc, actual_epochs)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model, optimizer, epoch, val_acc, val_loss, checkpoint_dir):
    """Save a model checkpoint to checkpoint_dir/model.pth."""
    checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, checkpoint_path)


def _finalize_training(model, history, config, best_val_loss, best_val_acc, actual_epochs):
    """Save final model, history JSON, and training curves."""
    final_path = os.path.join(config['save_dir'], 'garbage_classifier.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to: {final_path}")

    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)

    plot_training_history(history, config['save_dir'])

    final_train_loss = history['train_loss'][-1] if history['train_loss'] else None
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else None

    print(f"\n{'='*50}")
    print(f"  Training Summary")
    print(f"{'='*50}")
    print(f"  Total epochs run: {actual_epochs}")
    print(f"  Final train loss: {final_train_loss:.4f}" if final_train_loss else "  Final train loss: N/A")
    print(f"  Final val loss:   {final_val_loss:.4f}" if final_val_loss else "  Final val loss:   N/A")
    print(f"  Best val loss:    {best_val_loss:.4f}")
    print(f"  Best val acc:     {best_val_acc:.2f}%")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_training_history(history, save_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Learning rate plot
    axes[2].plot(history['lr'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Training curves saved to: {os.path.join(save_dir, 'training_curves.png')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Garbage Classifier')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--save-dir', type=str, default='../saved_models',
                        help='Path to save models')
    parser.add_argument('--model-type', type=str, default='efficientnet',
                        choices=['cnn', 'transfer', 'resnet', 'mobilenet', 'efficientnet'],
                        help='Model architecture (default: efficientnet)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (reduced to 32 for EfficientNet-B2 memory)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Phase 2 learning rate (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (Phase 2 only for EfficientNet)')
    parser.add_argument('--lr-finder', action='store_true',
                        help='Run LR range test before training (Task 6)')

    args = parser.parse_args()

    config = {
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'weight_decay': 0.01,
        'test_size': 0.15,
        'val_size': 0.15,
        'num_workers': 0,   # 0 for macOS/Windows compatibility
        'run_lr_finder': args.lr_finder,
    }

    train(config)


if __name__ == "__main__":
    main()
