"""
Multi-Instance Learning Training for EEG Regression
===================================================

Preserves temporal variation while preventing data leakage.

Key concepts:
1. Each subject is a "bag" containing multiple segment "instances"
2. Model processes segments individually
3. Predictions are aggregated at subject level before computing loss
4. Subject-wise cross-validation ensures no leakage

Usage:
    python mil_training.py --data_path /path/to/mil_data --k_fold 5
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import pickle
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from models.cbramod import CBraMod
from einops.layers.torch import Rearrange


def parse_args():
    parser = argparse.ArgumentParser(description='MIL Training for EEG Regression')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to MIL data directory')
    parser.add_argument('--k_fold', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--aggregation', type=str, default='mean',
                       choices=['mean', 'max', 'attention'],
                       help='How to aggregate segment predictions')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--pretrained_path', type=str,
                       default='pretrained_weights/pretrained_weights.pth')
    parser.add_argument('--use_pretrained', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='./results_mil')
    
    return parser.parse_args()


class MILDataset(torch.utils.data.Dataset):
    """
    Dataset for Multi-Instance Learning
    
    Each sample is a subject (bag) containing multiple segments (instances)
    """
    def __init__(self, mil_data_list):
        """
        Args:
            mil_data_list: List of dicts with keys:
                - 'subject_id': int
                - 'segments': array (n_segments, C, S, P)
                - 'label': float
                - 'n_segments': int
        """
        self.data = mil_data_list
        self.n_subjects = len(mil_data_list)
    
    def __len__(self):
        return self.n_subjects
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        segments = torch.FloatTensor(item['segments'])  # (n_seg, C, S, P)
        label = torch.FloatTensor([item['label']])  # (1,)
        subject_id = item['subject_id']
        
        return segments, label, subject_id


def collate_fn_mil(batch):
    """
    Custom collate function for variable-length bags
    
    Returns:
        all_segments: (total_segments, C, S, P) - flattened
        labels: (batch_size, 1) - one per subject
        segment_counts: (batch_size,) - how many segments per subject
        subject_ids: (batch_size,) - subject identifiers
    """
    all_segments = []
    labels = []
    segment_counts = []
    subject_ids = []
    
    for segments, label, subject_id in batch:
        all_segments.append(segments)  # (n_seg, C, S, P)
        labels.append(label)
        segment_counts.append(len(segments))
        subject_ids.append(subject_id)
    
    # Concatenate all segments
    all_segments = torch.cat(all_segments, dim=0)  # (total_segments, C, S, P)
    labels = torch.stack(labels)  # (batch_size, 1)
    segment_counts = torch.LongTensor(segment_counts)  # (batch_size,)
    subject_ids = torch.LongTensor(subject_ids)  # (batch_size,)
    
    return all_segments, labels, segment_counts, subject_ids


class AttentionAggregator(nn.Module):
    """Learnable attention-based aggregation"""
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, segment_features):
        """
        Args:
            segment_features: (n_segments, feature_dim)
        Returns:
            aggregated: (feature_dim,)
        """
        # Compute attention weights
        attn_weights = self.attention(segment_features)  # (n_segments, 1)
        attn_weights = torch.softmax(attn_weights, dim=0)  # (n_segments, 1)
        
        # Weighted sum
        aggregated = (segment_features * attn_weights).sum(dim=0)  # (feature_dim,)
        
        return aggregated


class MILRegressor(nn.Module):
    """
    Regressor with bag-level aggregation
    """
    def __init__(self, feature_dim, hidden_dim, dropout=0.2, aggregation='mean'):
        super().__init__()
        
        self.aggregation = aggregation
        
        # Flatten spatial dimensions
        self.flatten = Rearrange('b c s p -> b (c s p)')
        
        # Instance-level processing (optional)
        self.instance_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Aggregation
        if aggregation == 'attention':
            self.aggregator = AttentionAggregator(hidden_dim)
        
        # Bag-level prediction
        self.bag_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, segment_features, segment_counts):
        """
        Args:
            segment_features: (total_segments, C, S, P) - from backbone
            segment_counts: (batch_size,) - segments per subject
        
        Returns:
            predictions: (batch_size, 1) - one prediction per subject
        """
        # Flatten
        segment_features = self.flatten(segment_features)  # (total_segments, feature_dim)
        
        # Instance encoding
        segment_features = self.instance_encoder(segment_features)  # (total_segments, hidden_dim)
        
        # Aggregate by subject (bag)
        predictions = []
        start_idx = 0
        
        for count in segment_counts:
            end_idx = start_idx + count
            bag_segments = segment_features[start_idx:end_idx]  # (count, hidden_dim)
            
            # Aggregate this subject's segments
            if self.aggregation == 'mean':
                bag_repr = bag_segments.mean(dim=0)  # (hidden_dim,)
            elif self.aggregation == 'max':
                bag_repr = bag_segments.max(dim=0)[0]  # (hidden_dim,)
            elif self.aggregation == 'attention':
                bag_repr = self.aggregator(bag_segments)  # (hidden_dim,)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
            
            # Predict for this subject
            pred = self.bag_predictor(bag_repr)  # (1,)
            predictions.append(pred)
            
            start_idx = end_idx
        
        predictions = torch.stack(predictions)  # (batch_size, 1)
        
        return predictions


def train_epoch(model, regressor, loader, criterion, optimizer, device):
    """Train one epoch with MIL"""
    model.train()
    regressor.train()
    
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    
    for segments, labels, segment_counts, _ in pbar:
        segments = segments.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Extract features for all segments
        features = model(segments)  # (total_segments, C, S, P)
        
        # Aggregate and predict (one prediction per subject)
        predictions = regressor(features, segment_counts)  # (batch_size, 1)
        
        # Loss is computed at subject level!
        loss = criterion(predictions, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(regressor.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / n_batches


def validate(model, regressor, loader, device):
    """Validate with MIL"""
    model.eval()
    regressor.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for segments, labels, segment_counts, _ in loader:
            segments = segments.to(device)
            
            features = model(segments)
            predictions = regressor(features, segment_counts)
            
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    # Metrics
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)
    
    if len(preds) > 1:
        pearson_r, pearson_p = pearsonr(preds, labels)
        spearman_r, spearman_p = spearmanr(preds, labels)
    else:
        pearson_r, pearson_p = 0, 1
        spearman_r, spearman_p = 0, 1
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'n_subjects': len(preds)
    }


def main():
    args = parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*80)
    print("MULTI-INSTANCE LEARNING FOR EEG REGRESSION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Aggregation method: {args.aggregation}")
    
    # Load MIL data
    print("\n" + "="*80)
    print("LOADING MIL DATA")
    print("="*80)
    
    with open(os.path.join(args.data_path, 'mil_data.pkl'), 'rb') as f:
        mil_data = pickle.load(f)
    
    with open(os.path.join(args.data_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    print(f"✓ Loaded {len(mil_data)} subjects")
    print(f"  Total segments: {config['total_segments']}")
    print(f"  Avg segments/subject: {config['mean_segments_per_subject']:.1f}")
    
    # Get dimensions
    sample_segments = mil_data[0]['segments']
    n_channels = sample_segments.shape[1]
    time_segments = sample_segments.shape[2]
    points_per_patch = sample_segments.shape[3]
    feature_dim = n_channels * time_segments * points_per_patch
    
    print(f"\nData shape:")
    print(f"  Channels: {n_channels}")
    print(f"  Time segments: {time_segments}")
    print(f"  Points per patch: {points_per_patch}")
    print(f"  Feature dim: {feature_dim}")
    
    # K-fold CV
    print(f"\n{'='*80}")
    print(f"{args.k_fold}-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")
    
    kfold = KFold(n_splits=args.k_fold, shuffle=True, random_state=42)
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(mil_data)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{args.k_fold}")
        print(f"{'='*80}")
        
        # Split data
        train_data = [mil_data[i] for i in train_idx]
        test_data = [mil_data[i] for i in test_idx]
        
        # Create datasets
        train_dataset = MILDataset(train_data)
        test_dataset = MILDataset(test_data)
        
        # Create loaders with custom collate
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=4,  # Small batch since each "sample" is a whole subject
            shuffle=True,
            collate_fn=collate_fn_mil
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn_mil
        )
        
        print(f"Train: {len(train_data)} subjects")
        print(f"Test: {len(test_data)} subjects")
        
        # Create model
        model = CBraMod().to(device)
        
        if args.use_pretrained and os.path.exists(args.pretrained_path):
            model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
            print("✓ Loaded pretrained weights")
        
        model.proj_out = nn.Identity()
        
        # Create MIL regressor
        hidden_dim = 512
        regressor = MILRegressor(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=args.dropout,
            aggregation=args.aggregation
        ).to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': args.backbone_lr, 'weight_decay': args.weight_decay},
            {'params': regressor.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
        ])
        
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        print(f"\n{'='*60}")
        print("TRAINING")
        print(f"{'='*60}")
        
        best_r2 = -float('inf')
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, regressor, train_loader, criterion, optimizer, device)
            test_metrics = validate(model, regressor, test_loader, device)
            
            scheduler.step(test_metrics['rmse'])
            
            if test_metrics['r2'] > best_r2:
                best_r2 = test_metrics['r2']
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Test RMSE: {test_metrics['rmse']:6.3f} | "
                      f"R²: {test_metrics['r2']:6.3f} | "
                      f"Pearson: {test_metrics['pearson_r']:6.3f}")
        
        # Final evaluation
        final_metrics = validate(model, regressor, test_loader, device)
        fold_results.append(final_metrics)
        
        print(f"\nFold {fold_idx + 1} Final Results:")
        print(f"  RMSE:       {final_metrics['rmse']:.4f}")
        print(f"  MAE:        {final_metrics['mae']:.4f}")
        print(f"  R²:         {final_metrics['r2']:.4f}")
        print(f"  Pearson r:  {final_metrics['pearson_r']:.4f} (p={final_metrics['pearson_p']:.4e})")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"{args.k_fold}-FOLD CV SUMMARY")
    print(f"{'='*80}")
    
    r2_scores = [r['r2'] for r in fold_results]
    rmse_scores = [r['rmse'] for r in fold_results]
    pearson_scores = [r['pearson_r'] for r in fold_results]
    
    print(f"R²:         {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"RMSE:       {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Pearson r:  {np.mean(pearson_scores):.4f} ± {np.std(pearson_scores):.4f}")
    
    print(f"\n✓ MIL Training Complete!")
    print(f"\nKey properties:")
    print(f"  ✓ Temporal variation preserved (all segments used)")
    print(f"  ✓ No data leakage (subject-wise CV)")
    print(f"  ✓ Proper aggregation ({args.aggregation})")


if __name__ == '__main__':
    main()
