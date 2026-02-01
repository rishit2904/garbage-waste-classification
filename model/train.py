"""
Training Pipeline for Garbage Waste Classification

Features:
- Heavy data augmentation
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Mixed precision training
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import json

from cnn_model import get_model, GarbageClassifierCNN
from augmentation import get_training_augmentation, get_validation_augmentation


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


def load_dataset(data_dir, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load and split dataset into train, validation, and test sets
    
    Args:
        data_dir: Path to data directory
        test_size: Proportion for test set
        val_size: Proportion for validation set
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
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)
    
    if len(image_paths) == 0:
        print("No images found! Creating synthetic data for demo...")
        return None, None, None
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=test_size + val_size, 
        random_state=random_state, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + val_size),
        random_state=random_state, stratify=y_temp
    )
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"\nClass distribution (training):")
    for cls, count in Counter(y_train).items():
        print(f"  {classes[cls]}: {count}")
    
    # Create datasets
    train_transform = get_training_augmentation()
    val_transform = get_validation_augmentation()
    
    train_dataset = GarbageDataset(X_train, y_train, transform=train_transform)
    val_dataset = GarbageDataset(X_val, y_val, transform=val_transform)
    test_dataset = GarbageDataset(X_test, y_test, transform=val_transform)
    
    return train_dataset, val_dataset, test_dataset


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
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


def train(config):
    """
    Main training function
    
    Args:
        config: Training configuration dictionary
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_dataset(
        config['data_dir'],
        test_size=config.get('test_size', 0.2),
        val_size=config.get('val_size', 0.1)
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
        model = get_model(config.get('model_type', 'cnn'))
        os.makedirs(config['save_dir'], exist_ok=True)
        save_path = os.path.join(config['save_dir'], 'garbage_classifier.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Untrained model saved to: {save_path}")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model
    model = get_model(config.get('model_type', 'cnn'))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 10),
        restore_best_weights=True
    )
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"  -> New best model saved (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
        print(f"  {early_stopping.status}")
    
    # Save final model
    final_path = os.path.join(config['save_dir'], 'garbage_classifier.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    # Save training history
    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # Plot training curves
    plot_training_history(history, config['save_dir'])
    
    # Evaluate on test set
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4)
        )
        test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")


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


def main():
    parser = argparse.ArgumentParser(description='Train Garbage Classifier')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--save-dir', type=str, default='../saved_models',
                        help='Path to save models')
    parser.add_argument('--model-type', type=str, default='cnn',
                        choices=['cnn', 'resnet'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
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
        'num_workers': 0  # Set to 0 for compatibility
    }
    
    train(config)


if __name__ == "__main__":
    main()
