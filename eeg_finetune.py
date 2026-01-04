"""
Fine-tuning Script for CBraMod with EEG Data
=============================================

File: CBraMod/eeg_finetune.py

This script should be placed in the root CBraMod directory.
It performs fine-tuning of the pretrained CBraMod model on your EEG classification task.

Usage:
    python eeg_finetune.py --data_path /path/to/preprocessed/data --epochs 30

Arguments:
    --data_path: Path to directory containing X_train.npy, y_train.npy, etc.
    --pretrained_path: Path to pretrained CBraMod weights
    --batch_size: Batch size for training
    --lr: Learning rate for classifier
    --backbone_lr: Learning rate for pretrained backbone (usually 10x smaller)
    --epochs: Number of training epochs
    --device: Device to use (cuda/mps/cpu)
    --save_dir: Directory to save results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from datetime import datetime
import json

# Import CBraMod
from models.cbramod import CBraMod
from einops.layers.torch import Rearrange

# Import custom data loader
from datasets.eeg_loader import load_eeg_data

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune CBraMod on custom EEG data')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--pretrained_path', type=str,
                       default='pretrained_weights/pretrained_weights.pth',
                       help='Path to pretrained CBraMod weights')
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate for classifier')
    parser.add_argument('--backbone_lr', type=float, default=0.0001,
                       help='Learning rate for backbone (fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate in classifier')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for training')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--print_freq', type=int, default=5,
                       help='Print frequency (epochs)')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    # Auto-generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f'cbramod_finetune_{timestamp}'
    
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_classifier(feature_dim, hidden_dim, n_classes, dropout=0.1):
    """
    Create classifier head for CBraMod
    
    Args:
        feature_dim: Input feature dimension (channels * segments * points_per_patch)
        hidden_dim: Hidden layer dimension
        n_classes: Number of output classes
        dropout: Dropout rate
    
    Returns:
        nn.Module: Classifier network
    """
    classifier = nn.Sequential(
        # Flatten CBraMod output
        Rearrange('b c s p -> b (c s p)'),
        
        # First hidden layer
        nn.Linear(feature_dim, hidden_dim),
        nn.ELU(),
        nn.Dropout(dropout),
        
        # Second hidden layer
        nn.Linear(hidden_dim, hidden_dim // 4),
        nn.ELU(),
        nn.Dropout(dropout),
        
        # Output layer
        nn.Linear(hidden_dim // 4, n_classes),
    )
    
    return classifier


def train_epoch(model, classifier, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    classifier.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (TWO-STAGE for fine-tuning!)
        features = model(inputs)        # Extract features with CBraMod backbone
        outputs = classifier(features)   # Classify with task-specific head
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, classifier, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    classifier.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            features = model(inputs)
            outputs = classifier(features)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Get predictions
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Statistics
            total_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    save_path = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*80)
    print("CBRAMOD FINE-TUNING")
    print("="*80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {args.device}")
    print(f"Save path: {save_path}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    train_loader, test_loader, data_info = load_eeg_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cpu')
    )
    
    print(f"\nDataset info:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    
    # Setup device
    device = torch.device(args.device)
    
    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    model = CBraMod().to(device)
    
    if args.use_pretrained and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from: {args.pretrained_path}")
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        print("✓ Pretrained weights loaded")
    else:
        print("Training from scratch (no pretrained weights)")
    
    # Replace output projection with identity
    model.proj_out = nn.Identity()
    print("✓ Replaced output projection with Identity")
    
    # Create classifier
    feature_dim = data_info['feature_dim']
    hidden_dim = 4 * data_info['points_per_patch']
    n_classes = data_info['n_classes']
    
    classifier = create_classifier(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        n_classes=n_classes,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    backbone_params = sum(p.numel() for p in model.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    total_params = backbone_params + classifier_params
    
    print(f"\nModel architecture:")
    print(f"  Backbone parameters:   {backbone_params:,}")
    print(f"  Classifier parameters: {classifier_params:,}")
    print(f"  Total parameters:      {total_params:,}")
    print(f"\nClassifier structure:")
    print(f"  Input: {feature_dim:,} features")
    print(f"  Hidden: {hidden_dim:,} → {hidden_dim // 4:,}")
    print(f"  Output: {n_classes} classes")
    
    # Setup optimizer with differential learning rates
    print("\n" + "="*80)
    print("SETTING UP TRAINING")
    print("="*80)
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.backbone_lr, 'weight_decay': args.weight_decay},
        {'params': classifier.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
    ])
    
    print(f"Optimizer: Adam")
    print(f"  Backbone LR:    {args.backbone_lr} (fine-tuning)")
    print(f"  Classifier LR:  {args.lr} (training from scratch)")
    print(f"  Weight decay:   {args.weight_decay}")
    
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    print(f"Loss: CrossEntropyLoss")
    print(f"Scheduler: ReduceLROnPlateau (patience=5)")
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, classifier, train_loader, criterion, optimizer, device
        )
        
        # Validate
        test_loss, test_acc, test_labels, test_preds, test_probs = validate(
            model, classifier, test_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(test_acc)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'data_info': data_info,
                'args': vars(args)
            }, os.path.join(save_path, 'best_model.pth'))
        
        # Print progress
        if (epoch + 1) % args.print_freq == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} │ "
                  f"Train Loss: {train_loss:.4f} │ Train Acc: {train_acc:6.2f}% │ "
                  f"Test Loss: {test_loss:.4f} │ Test Acc: {test_acc:6.2f}%"
                  f"{' ← BEST!' if test_acc == best_acc else ''}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # Final validation
    _, final_acc, final_labels, final_preds, final_probs = validate(
        model, classifier, test_loader, criterion, device
    )
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds,
                              target_names=['Incorrect (200)', 'Correct (201)'],
                              digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # ROC AUC
    if len(np.unique(final_labels)) == 2:
        roc_auc = roc_auc_score(final_labels, final_probs)
        print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Save results
    results = {
        'best_acc': best_acc,
        'final_acc': final_acc,
        'history': history,
        'classification_report': classification_report(
            final_labels, final_preds, output_dict=True
        ),
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {save_path}")
    print(f"✓ Best model saved as: best_model.pth")


if __name__ == '__main__':
    main()
