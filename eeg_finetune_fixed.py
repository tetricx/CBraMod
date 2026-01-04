import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from datetime import datetime
import json
from timeit import default_timer as timer

from models.cbramod import CBraMod
from einops.layers.torch import Rearrange
from datasets.eeg_loader import load_eeg_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve, auc as pr_auc_score
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune CBraMod (Paper Table 6)')
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64,  # Paper: 64
                       help='Batch size (paper: 64)')
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--pretrained_path', type=str,
                       default='pretrained_weights/pretrained_weights.pth')
    parser.add_argument('--use_pretrained', action='store_true', default=True)
    
    parser.add_argument('--epochs', type=int, default=50,  # Paper: 50
                       help='Number of epochs (paper: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,  # Paper: 1e-4 (SAME for both)
                       help='Learning rate (paper: 1e-4, same for backbone and classifier)')
    parser.add_argument('--weight_decay', type=float, default=5e-2,  # Paper: 5e-2
                       help='Weight decay (paper: 5e-2)')
    parser.add_argument('--dropout', type=float, default=0.1,  # Paper: 0.1
                       help='Dropout rate (paper: 0.1)')
    parser.add_argument('--clip_value', type=float, default=1.0,  # Paper: 1.0
                       help='Gradient clipping value (paper: 1.0)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f'cbramod_paper_{timestamp}'
    
    return args


def create_classifier(feature_dim, hidden_dim, n_classes, dropout=0.1):
    """Create classifier matching CBraMod style"""
    classifier = nn.Sequential(
        Rearrange('b c s p -> b (c s p)'),
        nn.Linear(feature_dim, hidden_dim),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 4),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 4, n_classes),
    )
    return classifier


def get_metrics_for_binaryclass(model, classifier, data_loader, device):
    """Calculate metrics like CBraMod does"""
    model.eval()
    classifier.eval()
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            features = model(inputs)
            outputs = classifier(features)
            probs = F.softmax(outputs, dim=1)
            
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    
    if len(np.unique(all_labels)) == 2:
        roc_auc = roc_auc_score(all_labels, all_probs)
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = pr_auc_score(recall, precision)
    else:
        roc_auc = 0.0
        pr_auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, pr_auc, roc_auc, cm


def plot_confusion_matrix(cm, save_path, class_names=None):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")


def plot_training_curves(history, save_path):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history['val_acc'], 'g-', linewidth=2, label='Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: ROC-AUC
    axes[1, 0].plot(epochs, history['val_roc_auc'], 'r-', linewidth=2, label='ROC-AUC')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('ROC-AUC', fontsize=11)
    axes[1, 0].set_title('Validation ROC-AUC', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: PR-AUC
    axes[1, 1].plot(epochs, history['val_pr_auc'], 'm-', linewidth=2, label='PR-AUC')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('PR-AUC', fontsize=11)
    axes[1, 1].set_title('Validation PR-AUC', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved: {save_path}")


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    save_path = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*80)
    print("CBRAMOD FINE-TUNING (Paper Table 6 - ICLR 2025)")
    print("="*80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {args.device}")
    print(f"\nPaper Hyperparameters (Table 6):")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr:.0e} (same for backbone & classifier)")
    print(f"  Weight decay:  {args.weight_decay:.0e}")
    print(f"  Dropout:       {args.dropout}")
    print(f"  Optimizer:     AdamW")
    print(f"  Scheduler:     CosineAnnealingLR")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    train_loader, val_loader, data_info = load_eeg_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cpu')
    )
    
    print(f"\nDataset info:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    
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
    
    model.proj_out = nn.Identity()

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
    
    backbone_params = sum(p.numel() for p in model.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    
    print(f"\nModel:")
    print(f"  Backbone:   {backbone_params:,} parameters")
    print(f"  Classifier: {classifier_params:,} parameters")
    print(f"  Total:      {backbone_params + classifier_params:,} parameters")
    
    # Setup optimizer (Paper uses AdamW, not Adam)
    print("\n" + "="*80)
    print("SETUP TRAINING (Paper Table 6)")
    print("="*80)
    
    # CRITICAL: Paper uses AdamW with SAME learning rate for both
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': (args.lr)*0.1},
        {'params': classifier.parameters(), 'lr': args.lr}  # SAME LR!
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # CRITICAL: Paper uses CosineAnnealingLR (not OneCycleLR)
    # Stepped ONCE per epoch, not per batch!
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )
    
    print(f"Optimizer: AdamW (paper-compliant)")
    print(f"  Learning rate: {args.lr:.0e} (same for backbone & classifier)")
    print(f"  Weight decay:  {args.weight_decay:.0e}")
    print(f"  Adam beta:     (0.9, 0.999)")
    print(f"  Adam epsilon:  1e-8")
    print(f"\nScheduler: CosineAnnealingLR (paper-compliant)")
    print(f"  T_max:         {args.epochs} epochs")
    print(f"  Min LR:        1e-6")
    print(f"  Step:          Once per epoch (not per batch!)")
    print(f"\nOther:")
    print(f"  Gradient clip: {args.clip_value}")
    
    # Initialize history tracking
    history = {
        'train_loss': [],
        'val_acc': [],
        'val_pr_auc': [],
        'val_roc_auc': []
    }
    
    # Training loop (matching CBraMod's structure)
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    acc_best = 0
    roc_auc_best = 0
    pr_auc_best = 0
    cm_best = None
    best_model_states = None
    best_epoch = 0
    
    for epoch in range(args.epochs):
        model.train()
        classifier.train()
        
        start_time = timer()
        losses = []
        
        # Training loop with tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            features = model(x)
            pred = classifier(features)
            loss = criterion(pred, y)
            
            # Backward pass
            loss.backward()
            losses.append(loss.item())
            
            # Gradient clipping (like CBraMod)
            if args.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.clip_value)
            
            optimizer.step()
            # Note: Scheduler is stepped ONCE per epoch (see below)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Get current learning rates
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate average training loss
        avg_train_loss = np.mean(losses)
        
        # Validation (like CBraMod)
        with torch.no_grad():
            acc, pr_auc, roc_auc, cm = get_metrics_for_binaryclass(
                model, classifier, val_loader, device
            )
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_acc'].append(acc)
            history['val_pr_auc'].append(pr_auc)
            history['val_roc_auc'].append(roc_auc)
            
            elapsed_time = (timer() - start_time) / 60
            
            print(f"\nEpoch {epoch + 1}:")
            print(f"  Training Loss: {avg_train_loss:.5f}")
            print(f"  Val - Acc: {acc:.5f}, PR-AUC: {pr_auc:.5f}, ROC-AUC: {roc_auc:.5f}")
            print(f"  LR: {current_lr:.7f}")
            print(f"  Time: {elapsed_time:.2f} mins")
            print(f"  Confusion Matrix:")
            print(f"    {cm}")
            
            # Save best model based on accuracy (like CBraMod)
            if acc > acc_best:
                print("  → acc increasing... saving weights!")
                best_epoch = epoch + 1
                acc_best = acc
                pr_auc_best = pr_auc
                roc_auc_best = roc_auc
                cm_best = cm
                best_model_states = copy.deepcopy({
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict()
                })
        
        # Step scheduler ONCE per epoch (paper approach)
        scheduler.step()
    
    # Load best model and test
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    model.load_state_dict(best_model_states['model'])
    classifier.load_state_dict(best_model_states['classifier'])
    
    with torch.no_grad():
        acc, pr_auc, roc_auc, cm = get_metrics_for_binaryclass(
            model, classifier, val_loader, device
        )
        
        print(f"\nBest Model (Epoch {best_epoch}):")
        print(f"  Accuracy:  {acc:.5f}")
        print(f"  PR-AUC:    {pr_auc:.5f}")
        print(f"  ROC-AUC:   {roc_auc:.5f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Save model
        model_path = os.path.join(
            save_path,
            f"epoch{best_epoch}_acc_{acc:.5f}_pr_{pr_auc:.5f}_roc_{roc_auc:.5f}.pth"
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'metrics': {
                'acc': acc,
                'pr_auc': pr_auc,
                'roc_auc': roc_auc,
                'cm': cm.tolist()
            },
            'epoch': best_epoch,
            'args': vars(args),
            'history': history
        }, model_path)
        
        print(f"\n✓ Model saved: {model_path}")
    
    # Generate and save plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Plot confusion matrix
    cm_plot_path = os.path.join(save_path, 'confusion_matrix.png')
    class_names = [f'Class {i}' for i in range(n_classes)]
    plot_confusion_matrix(cm_best, cm_plot_path, class_names)
    
    # Plot training curves
    curves_plot_path = os.path.join(save_path, 'training_curves.png')
    plot_training_curves(history, curves_plot_path)
    
    # Save history as JSON
    history_path = os.path.join(save_path, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved: {history_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {save_path}")
    print(f"  - Model weights: {os.path.basename(model_path)}")
    print(f"  - Confusion matrix: confusion_matrix.png")
    print(f"  - Training curves: training_curves.png")
    print(f"  - Training history: training_history.json")


if __name__ == '__main__':
    main()