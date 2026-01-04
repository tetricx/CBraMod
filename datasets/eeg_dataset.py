"""
Custom Dataset for EEG Classification Task
===========================================

File: CBraMod/datasets/eeg_dataset.py

This file should be placed in the CBraMod/datasets/ directory.
It defines a dataset class compatible with CBraMod's architecture.

Task: Binary classification (Correct=201 vs Incorrect=200)
Format: (n_channels, time_segments, points_per_patch)
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class EEGDataset(Dataset):
    """
    Custom EEG Dataset for CBraMod fine-tuning
    
    Data format expected by CBraMod:
        - Input shape: (n_channels, time_segments, points_per_patch)
        - Example: (22, 4, 200) means 22 channels, 4 temporal segments, 200 points per segment
    
    Args:
        X (numpy.ndarray): EEG data of shape (n_samples, n_channels, time_segments, points_per_patch)
        y (numpy.ndarray): Labels of shape (n_samples,)
        transform (callable, optional): Optional transform to be applied on a sample
    """
    
    def __init__(self, X, y, transform=None):
        """
        Initialize the dataset
        
        Args:
            X: EEG data array (n_samples, n_channels, time_segments, points_per_patch)
            y: Label array (n_samples,)
            transform: Optional data augmentation/transformation
        """
        assert len(X) == len(y), "X and y must have the same length"
        assert X.ndim == 4, f"X must be 4D (n, c, s, p), got shape {X.shape}"
        
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
        # Store data statistics
        self.n_samples = len(X)
        self.n_channels = X.shape[1]
        self.time_segments = X.shape[2]
        self.points_per_patch = X.shape[3]
        self.n_classes = len(np.unique(y))
        
        print(f"Dataset created:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Channels: {self.n_channels}")
        print(f"  Time segments: {self.time_segments}")
        print(f"  Points per patch: {self.points_per_patch}")
        print(f"  Classes: {self.n_classes}")
    
    def __len__(self):
        """Return the total number of samples"""
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (eeg_data, label)
                - eeg_data: Tensor of shape (n_channels, time_segments, points_per_patch)
                - label: Scalar tensor
        """
        sample = self.X[idx]
        label = self.y[idx]
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
    
    def get_data_info(self):
        """
        Get dataset statistics
        
        Returns:
            dict: Dictionary containing dataset information
        """
        return {
            'n_samples': self.n_samples,
            'n_channels': self.n_channels,
            'time_segments': self.time_segments,
            'points_per_patch': self.points_per_patch,
            'n_classes': self.n_classes,
            'input_shape': (self.n_channels, self.time_segments, self.points_per_patch),
            'label_distribution': {
                int(i): int((self.y == i).sum()) 
                for i in torch.unique(self.y)
            }
        }


# Optional: Data augmentation transforms
class RandomTimeShift:
    """Randomly shift the time axis"""
    def __init__(self, max_shift=10):
        self.max_shift = max_shift
    
    def __call__(self, sample):
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        if shift > 0:
            sample = torch.cat([sample[:, :, -shift:], sample[:, :, :-shift]], dim=2)
        elif shift < 0:
            sample = torch.cat([sample[:, :, -shift:], sample[:, :, :-shift]], dim=2)
        return sample


class RandomAmplitudeScale:
    """Randomly scale the amplitude"""
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range
    
    def __call__(self, sample):
        scale = torch.FloatTensor(1).uniform_(*self.scale_range)
        return sample * scale


class AddGaussianNoise:
    """Add Gaussian noise to the signal"""
    def __init__(self, std=0.01):
        self.std = std
    
    def __call__(self, sample):
        noise = torch.randn_like(sample) * self.std
        return sample + noise


if __name__ == '__main__':
    # Test the dataset
    print("Testing EEGDataset...")
    
    # Create dummy data
    n_samples = 100
    n_channels = 22
    time_segments = 4
    points_per_patch = 200
    n_classes = 2
    
    X_dummy = np.random.randn(n_samples, n_channels, time_segments, points_per_patch)
    y_dummy = np.random.randint(0, n_classes, n_samples)
    
    # Create dataset
    dataset = EEGDataset(X_dummy, y_dummy)
    
    # Test dataset access
    print(f"\nTesting dataset access...")
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Label: {label}")
    
    # Get data info
    print(f"\nDataset info:")
    info = dataset.get_data_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with augmentation
    print(f"\nTesting with augmentation...")
    transform = torch.nn.Sequential(
        RandomTimeShift(max_shift=5),
        RandomAmplitudeScale(scale_range=(0.95, 1.05)),
        AddGaussianNoise(std=0.005)
    )
    dataset_aug = EEGDataset(X_dummy, y_dummy, transform=transform)
    sample_aug, _ = dataset_aug[0]
    print(f"Augmented sample shape: {sample_aug.shape}")
    
    print("\n✓ All tests passed!")
