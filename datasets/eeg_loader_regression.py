"""
Fixed EEG Data Loader for Regression - Subject-wise Split
==========================================================
Fixes:
1. Proper subject-wise data organization
2. No data leakage between train/val
3. Efficient batching by subject
4. Returns data in correct format for training
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import json
import os


class SubjectEEGDataset(Dataset):
    """
    Dataset that returns ALL segments for a subject at once.
    This prevents data leakage and allows proper aggregation.
    """
    
    def __init__(self, subject_ids, X, y, subjects):
        """
        Args:
            subject_ids: List of subject IDs in this split
            X: Full array (n_windows, C, T, P)
            y: Full array (n_windows,) of scores
            subjects: Full array (n_windows,) of subject IDs
        """
        self.subject_ids = subject_ids
        self.X = X
        self.y = y
        self.subjects = subjects
        
        # Pre-compute indices for each subject (for efficiency)
        self.subject_indices = {}
        for subj_id in subject_ids:
            self.subject_indices[subj_id] = np.where(subjects == subj_id)[0]
    
    def __len__(self):
        return len(self.subject_ids)
    
    def __getitem__(self, idx):
        """
        Returns:
            segments: (n_segments, C, T, P) - all segments for this subject
            score: (1,) - the subject's Raven score
        """
        subject_id = self.subject_ids[idx]
        indices = self.subject_indices[subject_id]
        
        # Get all windows for this subject
        segments = torch.FloatTensor(self.X[indices])
        score = torch.FloatTensor([self.y[indices[0]]])  # All same score
        
        return segments, score


def collate_subject_batch(batch):
    """
    Custom collate function that handles variable-length segment lists.
    
    Args:
        batch: List of (segments, score) tuples
    
    Returns:
        segments_list: List of (n_segments_i, C, T, P) tensors
        scores: (batch_size,) tensor
    """
    segments_list = [item[0] for item in batch]
    scores = torch.stack([item[1] for item in batch]).squeeze()
    
    return segments_list, scores


def load_eeg_data_regression(data_path, batch_size=4, num_workers=4, 
                            pin_memory=True, cv_fold=None, n_folds=5, 
                            seed=42):
    """
    Load EEG data with proper subject-wise splitting.
    
    Args:
        data_path: Path to directory with X.npy, y.npy, subjects.npy, config.json
        batch_size: Number of SUBJECTS per batch
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        cv_fold: Which fold to use (0 to n_folds-1), None for simple split
        n_folds: Number of CV folds
        seed: Random seed
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        data_info: Dict with dataset information
    """
    
    print(f"Loading data from: {data_path}")
    
    # Load data
    X = np.load(os.path.join(data_path, 'X.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    subjects = np.load(os.path.join(data_path, 'subjects.npy'))
    
    with open(os.path.join(data_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  subjects shape: {subjects.shape}")
    
    # Get unique subjects
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    print(f"\nDataset statistics:")
    print(f"  Total windows: {len(X)}")
    print(f"  Unique subjects: {n_subjects}")
    print(f"  Windows per subject: {len(X) / n_subjects:.1f} (avg)")
    
    # Verify no overlap in scores per subject
    for subj in unique_subjects:
        subj_scores = y[subjects == subj]
        assert np.all(subj_scores == subj_scores[0]), \
            f"Subject {subj} has inconsistent scores!"
    
    # Subject-wise split
    np.random.seed(seed)
    shuffled_subjects = unique_subjects.copy()
    np.random.shuffle(shuffled_subjects)
    
    if cv_fold is not None:
        # K-fold cross-validation
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(kfold.split(shuffled_subjects))
        train_idx, val_idx = folds[cv_fold]
        train_subjects = shuffled_subjects[train_idx]
        val_subjects = shuffled_subjects[val_idx]
        print(f"\nUsing CV fold {cv_fold}/{n_folds}")
    else:
        # Simple 80/20 split
        split_idx = int(0.8 * n_subjects)
        train_subjects = shuffled_subjects[:split_idx]
        val_subjects = shuffled_subjects[split_idx:]
        print(f"\nUsing 80/20 train/val split")
    
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Val subjects: {len(val_subjects)}")
    
    # Verify no overlap
    overlap = set(train_subjects) & set(val_subjects)
    assert len(overlap) == 0, f"Data leakage! {len(overlap)} subjects in both splits!"
    
    # Create datasets
    train_dataset = SubjectEEGDataset(train_subjects, X, y, subjects)
    val_dataset = SubjectEEGDataset(val_subjects, X, y, subjects)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_subject_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_subject_batch
    )
    
    # Prepare data info
    data_info = {
        'n_train_subjects': len(train_subjects),
        'n_val_subjects': len(val_subjects),
        'n_train_windows': sum(len(train_dataset.subject_indices[s]) 
                              for s in train_subjects),
        'n_val_windows': sum(len(val_dataset.subject_indices[s]) 
                            for s in val_subjects),
        'n_channels': X.shape[1],
        'time_segments': X.shape[2],
        'points_per_patch': X.shape[3],
        'feature_dim': X.shape[1] * X.shape[2] * X.shape[3],
        'target_mean': float(config.get('target_mean', np.mean(y))),
        'target_std': float(config.get('target_std', np.std(y))),
        'target_range': config.get('target_range', [float(np.min(y)), float(np.max(y))]),
        'input_shape': X.shape[1:],
    }
    
    print(f"\nData info:")
    print(f"  Train: {data_info['n_train_subjects']} subjects, "
          f"{data_info['n_train_windows']} windows")
    print(f"  Val: {data_info['n_val_subjects']} subjects, "
          f"{data_info['n_val_windows']} windows")
    print(f"  Input shape: {data_info['input_shape']}")
    print(f"  Target mean: {data_info['target_mean']:.2f}")
    print(f"  Target std: {data_info['target_std']:.2f}")
    
    return train_loader, val_loader, data_info