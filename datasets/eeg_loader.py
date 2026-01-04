"""
Data Loader for EEG Classification Task - FIXED VERSION
========================================================

File: CBraMod/datasets/eeg_loader.py

This version handles the naming inconsistency between:
- Classification_preprocessing.py which uses 'n_patches'
- The training code which expects 'time_segments'

Usage:
    from datasets.eeg_loader import load_eeg_data
    
    train_loader, test_loader, data_info = load_eeg_data(
        data_path='/path/to/preprocessed/data',
        batch_size=32
    )
"""

import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, Dataset


class EEGDataset(Dataset):
    """Simple EEG Dataset"""
    
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x, y


def load_eeg_data(data_path, batch_size=32, num_workers=4, pin_memory=True,
                  use_augmentation=False, shuffle_train=True):
    """
    Load preprocessed EEG data and create PyTorch DataLoaders
    
    Args:
        data_path (str): Path to directory containing:
            - X_train.npy: Training data
            - y_train.npy: Training labels
            - X_test.npy: Test data  
            - y_test.npy: Test labels
            - config.json: Dataset configuration
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Pin memory for faster GPU transfer
        use_augmentation (bool): Whether to apply data augmentation
        shuffle_train (bool): Whether to shuffle training data
    
    Returns:
        tuple: (train_loader, test_loader, data_info)
            - train_loader: DataLoader for training set
            - test_loader: DataLoader for test set
            - data_info: Dictionary with dataset information
    """
    
    print(f"Loading data from: {data_path}")
    
    # Check if required files exist
    required_files = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']
    for file in required_files:
        filepath = os.path.join(data_path, file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")
    
    # Load data
    print("Loading numpy arrays...")
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Load config if available
    config_path = os.path.join(data_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\nConfiguration loaded:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # CRITICAL FIX: Handle naming inconsistency between preprocessing and training
        # Classification_preprocessing.py uses 'n_patches'
        # Training code expects 'time_segments'
        if 'n_patches' in config and 'time_segments' not in config:
            config['time_segments'] = config['n_patches']
            print(f"  ✓ Fixed: Using 'n_patches' ({config['n_patches']}) as 'time_segments'")
        
        # Ensure all required keys exist
        if 'time_segments' not in config:
            config['time_segments'] = X_train.shape[2]
            print(f"  ✓ Inferred: time_segments = {config['time_segments']}")
        
        if 'n_train' not in config:
            config['n_train'] = len(X_train)
        if 'n_test' not in config:
            config['n_test'] = len(X_test)
            
    else:
        print("\nWarning: config.json not found, inferring from data...")
        config = {
            'n_channels': X_train.shape[1],
            'time_segments': X_train.shape[2],
            'points_per_patch': X_train.shape[3],
            'n_classes': len(np.unique(y_train)),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    # Data augmentation (optional)
    transform = None
    if use_augmentation:
        print("\n✓ Data augmentation enabled")
        # Add your augmentation transforms here if needed
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EEGDataset(X_train, y_train, transform=transform)
    test_dataset = EEGDataset(X_test, y_test, transform=None)  # No augmentation for test
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Keep all samples
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Prepare data info dictionary
    data_info = {
        'n_channels': config['n_channels'],
        'time_segments': config['time_segments'],  # Now guaranteed to exist
        'points_per_patch': config['points_per_patch'],
        'n_classes': config['n_classes'],
        'n_train': config['n_train'],
        'n_test': config['n_test'],
        'batch_size': batch_size,
        'train_batches': len(train_loader),
        'test_batches': len(test_loader),
        'feature_dim': config['n_channels'] * config['time_segments'] * config['points_per_patch']
    }
    
    print("\n✓ Data loading complete!")
    print(f"\nData info summary:")
    print(f"  Shape: ({config['n_channels']} channels, {config['time_segments']} patches, {config['points_per_patch']} points)")
    print(f"  Feature dim: {data_info['feature_dim']}")
    print(f"  Classes: {config['n_classes']}")
    
    return train_loader, test_loader, data_info


if __name__ == '__main__':
    # Test the loader
    print("Testing data loader...")
    
    # Create dummy data for testing
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    print(f"\nCreating dummy data in: {temp_dir}")
    X_train_dummy = np.random.randn(80, 19, 4, 200).astype(np.float32)
    y_train_dummy = np.random.randint(0, 2, 80)
    X_test_dummy = np.random.randn(20, 19, 4, 200).astype(np.float32)
    y_test_dummy = np.random.randint(0, 2, 20)
    
    np.save(os.path.join(temp_dir, 'X_train.npy'), X_train_dummy)
    np.save(os.path.join(temp_dir, 'y_train.npy'), y_train_dummy)
    np.save(os.path.join(temp_dir, 'X_test.npy'), X_test_dummy)
    np.save(os.path.join(temp_dir, 'y_test.npy'), y_test_dummy)
    
    # Test with old naming convention (n_patches)
    config_dummy = {
        'format': 'official_cbramod',
        'n_channels': 19,
        'n_patches': 4,  # OLD NAMING
        'points_per_patch': 200,
        'n_classes': 2,
        'n_train': 80,
        'n_test': 20
    }
    with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
        json.dump(config_dummy, f)
    
    # Test loading
    train_loader, test_loader, data_info = load_eeg_data(
        temp_dir, batch_size=16
    )
    
    print("\n✓ Data loaders created successfully!")
    print(f"\nData info:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    
    # Verify time_segments was correctly set
    assert data_info['time_segments'] == 4, "time_segments should be 4!"
    print("\n✓ Naming inconsistency fix verified!")
    
    # Test iteration
    print("\nTesting batch iteration...")
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"  Batch {batch_idx}: data shape = {data.shape}, labels shape = {labels.shape}")
        if batch_idx >= 2:  # Only show first 3 batches
            break
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n✓ All tests passed!")