"""
DEBUGGING VERSION: CBraMod Fine-tuning with Better Diagnostics
===============================================================

Adds extensive debugging and fixes for overfitting issues
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from datetime import datetime
import json
from timeit import default_timer as timer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy

from models.cbramod import CBraMod
from einops.layers.torch import Rearrange


class EEGRegressionDataset(Dataset):
    def __init__(self, X, y, normalize_targets=True):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        # Normalize targets for stable training
        if normalize_targets:
            self.y_mean = self.y.mean()
            self.y_std = self.y.std()
            self.y = (self.y - self.y_mean) / (self.y_std + 1e-8)
            print(f"  Target normalization: mean={self.y_mean:.2f}, std={self.y_std:.2f}")
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def denormalize(self, y_normalized):
            """Convert normalized predictions back to original scale"""
            # FIX: Convert PyTorch tensors to Python floats using .item()
            std = self.y_std.item() if isinstance(self.y_std, torch.Tensor) else self.y_std
            mean = self.y_mean.item() if isinstance(self.y_mean, torch.Tensor) else self.y_mean
            
            return y_normalized * std + mean


class ImprovedRegressionHead(nn.Module):
    """Simpler, more stable regression head"""
    
    def __init__(self, input_shape, dropout=0.4):
        super().__init__()
        
        C, n, hidden_dim = input_shape
        feature_dim = C * n * hidden_dim
        
        print(f"  Regression head:")
        print(f"    Input dim: {feature_dim}")
        print(f"    Dropout: {dropout}")
        
        # Simpler architecture
        self.net = nn.Sequential(
            Rearrange('b c n h -> b (c n h)'),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def validate_data(X, y, subjects):
    """Comprehensive data validation"""
    print(f"\n{'='*80}")
    print("DATA VALIDATION")
    print(f"{'='*80}")
    
    # Check for NaN/Inf
    print(f"\nNaN/Inf checks:")
    print(f"  X has NaN: {np.isnan(X).any()}")
    print(f"  X has Inf: {np.isinf(X).any()}")
    print(f"  y has NaN: {np.isnan(y).any()}")
    print(f"  y has Inf: {np.isinf(y).any()}")
    
    if np.isnan(X).any() or np.isinf(X).any():
        print("  ⚠️  WARNING: X contains NaN or Inf!")
        n_bad = np.sum(np.isnan(X) | np.isinf(X))
        print(f"  Bad values: {n_bad}")
    
    # Check X normalization
    print(f"\nX statistics:")
    print(f"  Shape: {X.shape}")
    print(f"  Mean:  {X.mean():.4f}")
    print(f"  Std:   {X.std():.4f}")
    print(f"  Min:   {X.min():.2f}")
    print(f"  Max:   {X.max():.2f}")
    
    # Per-sample statistics
    sample_means = X.reshape(len(X), -1).mean(axis=1)
    sample_stds = X.reshape(len(X), -1).std(axis=1)
    print(f"\nPer-sample statistics:")
    print(f"  Mean range: [{sample_means.min():.4f}, {sample_means.max():.4f}]")
    print(f"  Std range:  [{sample_stds.min():.4f}, {sample_stds.max():.4f}]")
    
    # Check y distribution
    print(f"\ny (target) statistics:")
    print(f"  Shape: {y.shape}")
    print(f"  Mean:  {y.mean():.2f}")
    print(f"  Std:   {y.std():.2f}")
    print(f"  Min:   {y.min():.1f}")
    print(f"  Max:   {y.max():.1f}")
    
    # Check for constant predictions
    if y.std() < 0.1:
        print(f"  ⚠️  WARNING: Very low target variance!")
    
    # Subject distribution
    unique_subjects = np.unique(subjects)
    samples_per_subject = [np.sum(subjects == s) for s in unique_subjects]
    print(f"\nSubject distribution:")
    print(f"  Unique subjects: {len(unique_subjects)}")
    print(f"  Samples/subject: {np.mean(samples_per_subject):.1f} ± {np.std(samples_per_subject):.1f}")
    print(f"  Min samples:     {np.min(samples_per_subject)}")
    print(f"  Max samples:     {np.max(samples_per_subject)}")


def load_data(data_path, cv_fold=None, n_folds=5, seed=42):
    """Load and split data with validation"""
    
    X = np.load(os.path.join(data_path, 'X.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    subjects = np.load(os.path.join(data_path, 'subjects.npy'))
    
    with open(os.path.join(data_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Validate data
    validate_data(X, y, subjects)
    
    # Subject-level split
    unique_subjects = np.unique(subjects)
    
    if cv_fold is not None:
        kfold = GroupKFold(n_splits=n_folds)
        splits = list(kfold.split(X, y, groups=subjects))
        train_idx, val_idx = splits[cv_fold]
        print(f"\n{'='*80}")
        print(f"USING CV FOLD {cv_fold}/{n_folds}")
        print(f"{'='*80}")
    else:
        np.random.seed(seed)
        np.random.shuffle(unique_subjects)
        n_train = int(0.8 * len(unique_subjects))
        train_subjects = set(unique_subjects[:n_train])
        train_idx = [i for i, s in enumerate(subjects) if s in train_subjects]
        val_idx = [i for i, s in enumerate(subjects) if s not in train_subjects]
        print(f"\n{'='*80}")
        print(f"USING 80/20 SPLIT")
        print(f"{'='*80}")
    
    # Check for data leakage
    train_subj_set = set(subjects[train_idx])
    val_subj_set = set(subjects[val_idx])
    overlap = train_subj_set & val_subj_set
    
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_idx)} samples from {len(train_subj_set)} subjects")
    print(f"  Val:   {len(val_idx)} samples from {len(val_subj_set)} subjects")
    print(f"  Overlap: {len(overlap)} subjects (MUST be 0!)")
    
    if len(overlap) > 0:
        print(f"  ⚠️  ERROR: Data leakage detected!")
        raise ValueError("Train and validation sets have overlapping subjects!")
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Check target distributions match
    print(f"\nTarget distribution check:")
    print(f"  Train mean: {y_train.mean():.2f} ± {y_train.std():.2f}")
    print(f"  Val mean:   {y_val.mean():.2f} ± {y_val.std():.2f}")
    
    return (X_train, y_train), (X_val, y_val), config


def evaluate(model, regressor, data_loader, criterion, device, dataset):
    """Evaluate with denormalization"""
    model.eval()
    regressor.eval()
    
    all_preds_norm = []
    all_targets_norm = []
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            features = model(X_batch)
            preds = regressor(features)
            
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(X_batch)
            
            all_preds_norm.extend(preds.cpu().numpy().tolist())
            all_targets_norm.extend(y_batch.cpu().numpy().tolist())
    
    all_preds_norm = np.array(all_preds_norm)
    all_targets_norm = np.array(all_targets_norm)
    
    # Denormalize
    all_preds = dataset.denormalize(all_preds_norm)
    all_targets = dataset.denormalize(all_targets_norm)
    
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    avg_loss = total_loss / len(data_loader.dataset)
    
    return {
        'loss': avg_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)  # Smaller default
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--pretrained_path', type=str,
                       default='pretrained_weights/pretrained_weights.pth')
    parser.add_argument('--use_pretrained', action='store_true', default=True)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)  # Lower default
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.4)  # Higher default
    parser.add_argument('--clip_value', type=float, default=0.5)  # Lower default
    
    parser.add_argument('--normalize_targets', action='store_true', default=True)
    parser.add_argument('--loss_fn', type=str, default='huber',
                       choices=['mse', 'l1', 'huber', 'smooth_l1'])
    
    parser.add_argument('--cv_fold', type=int, default=None)
    parser.add_argument('--n_folds', type=int, default=5)
    
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str, default='./results_debug')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fold_str = f"_fold{args.cv_fold}" if args.cv_fold is not None else ""
    exp_name = f'debug{fold_str}_{timestamp}'
    save_path = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    print("="*80)
    print("DEBUGGING CBRAMOD TRAINING")
    print("="*80)
    print(f"Experiment: {exp_name}")
    print(f"Device:     {args.device}")
    
    # Load data with validation
    train_data, val_data, config = load_data(
        args.data_path, args.cv_fold, args.n_folds, args.seed
    )
    
    train_dataset = EEGRegressionDataset(*train_data, normalize_targets=args.normalize_targets)
    val_dataset = EEGRegressionDataset(*val_data, normalize_targets=args.normalize_targets)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == 'cuda')
    )
    
    # Model
    print(f"\n{'='*80}")
    print("MODEL SETUP")
    print(f"{'='*80}")
    
    model = CBraMod().to(device)
    for param in model.parameters():
            param.requires_grad = False
    if args.use_pretrained and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights...")
        state_dict = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("  ✓ Loaded")
    
    model.proj_out = nn.Identity()
    
    # Get output shape
    with torch.no_grad():
        dummy = torch.randn(1, config['n_channels'], config['n_patches'], config['patch_size']).to(device)
        output = model(dummy)
        output_shape = output.shape[1:]
    
    print(f"\nI/O shapes:")
    print(f"  Input:  (batch, {config['n_channels']}, {config['n_patches']}, {config['patch_size']})")
    print(f"  Output: (batch, {output_shape[0]}, {output_shape[1]}, {output_shape[2]})")
    
    regressor = ImprovedRegressionHead(
        input_shape=output_shape,
        dropout=args.dropout
    ).to(device)
    
    # Training setup
    print(f"\n{'='*80}")
    print("TRAINING SETUP")
    print(f"{'='*80}")
    
    # Much lower learning rates
    backbone_lr = args.lr * 0.1
    head_lr = args.lr
    
    print(f"Learning rates:")
    print(f"  Backbone: {backbone_lr:.2e}")
    print(f"  Head:     {head_lr:.2e}")
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': backbone_lr},
        {'params': regressor.parameters(), 'lr': head_lr}
    ], weight_decay=args.weight_decay)
    
    # Loss
    if args.loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_fn == 'l1':
        criterion = nn.L1Loss()
    elif args.loss_fn == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.HuberLoss(delta=1.0)
    
    # Scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-8
    )
    
    print(f"\nOptimizer:  AdamW")
    print(f"Loss:       {args.loss_fn}")
    print(f"Scheduler:  CosineAnnealingWarmRestarts")
    print(f"Clip value: {args.clip_value}")
    
    # Training
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}")
    
    best_mae = float('inf')
    best_model_state = None
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        regressor.train()
        
        losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            features = model(X_batch)
            preds = regressor(features)
            loss = criterion(preds, y_batch)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  NaN/Inf loss at epoch {epoch+1}")
                print(f"  Preds: {preds[:5]}")
                print(f"  Targets: {y_batch[:5]}")
                continue
            
            loss.backward()
            losses.append(loss.item())
            
            if args.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(regressor.parameters()),
                    args.clip_value
                )
            
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if len(losses) == 0:
            print("No valid losses!")
            break
        
        # Validation
        metrics = evaluate(model, regressor, val_loader, criterion, device, val_dataset)
        scheduler.step()
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Train Loss: {np.mean(losses):.4f}")
        print(f"  Val Loss:   {metrics['loss']:.4f}")
        print(f"  Val MAE:    {metrics['mae']:.3f}")
        print(f"  Val RMSE:   {metrics['rmse']:.3f}")
        print(f"  Val R²:     {metrics['r2']:.3f}")
        
        # Check for divergence
        if metrics['r2'] < -2.0:
            print("  ⚠️  Model diverging! (R² < -2)")
        
        if metrics['mae'] < best_mae:
            print("  → New best!")
            best_mae = metrics['mae']
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = {
                'model': copy.deepcopy(model.state_dict()),
                'regressor': copy.deepcopy(regressor.state_dict()),
                'metrics': metrics
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    # Final results
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    
    if best_model_state is not None:
        metrics = best_model_state['metrics']
        print(f"\nBest Model (Epoch {best_epoch}):")
        print(f"  MAE:  {metrics['mae']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  R²:   {metrics['r2']:.3f}")
        
        # Save
        model_path = os.path.join(save_path, f"best_model.pth")
        torch.save({
            'model': best_model_state['model'],
            'regressor': best_model_state['regressor'],
            'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in metrics.items()},
            'epoch': best_epoch,
            'config': config,
            'args': vars(args)
        }, model_path)
        
        print(f"\n✓ Saved: {model_path}")
    else:
        print("\n⚠️  No best model saved!")


if __name__ == '__main__':
    main()