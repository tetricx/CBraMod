"""
Custom Dataset for EEG Regression Task (Raven Score Prediction)
================================================================

File: CBraMod/datasets/eeg_dataset_regression.py

This file should be placed in the CBraMod/datasets/ directory.
It defines a dataset class compatible with CBraMod's architecture for regression tasks.

Task: Regression (Predict Raven score)
Format: (n_channels, time_segments, points_per_patch)
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class EEGRegressionDataset(Dataset):
    """
    Custom EEG Dataset for CBraMod fine-tuning on regression tasks
    
    Data format expected by CBraMod:
        - Input shape: (n_channels, time_segments, points_per_patch)
        - Example: (22, 4, 200) means 22 channels, 4 temporal segments, 200 points per segment
    
    Args:
        X (numpy.ndarray): EEG data of shape (n_samples, n_channels, time_segments, points_per_patch)
        y (numpy.ndarray): Regression targets (e.g., Raven scores) of shape (n_samples,)
        subjects (numpy.ndarray): Subject IDs of shape (n_samples,)
        transform (callable, optional): Optional transform to be applied on a sample
    """
    
    def __init__(self, X, y, subjects=None, transform=None):
        """
        Initialize the dataset
        
        Args:
            X: EEG data array (n_samples, n_channels, time_segments, points_per_patch)
            y: Regression targets (n_samples,) - e.g., Raven scores
            subjects: Subject IDs (n_samples,) - for subject-wise splits
            transform: Optional data augmentation/transformation
        """
        assert len(X) == len(y), "X and y must have the same length"
        assert X.ndim == 4, f"X must be 4D (n, c, s, p), got shape {X.shape}"
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)  # FloatTensor for regression
        
        if subjects is not None:
            assert len(subjects) == len(X), "subjects and X must have the same length"
            self.subjects = subjects
        else:
            self.subjects = np.arange(len(X))  # Default: each sample is its own subject
        
        self.transform = transform
        
        # Store data statistics
        self.n_samples = len(X)
        self.n_channels = X.shape[1]
        self.time_segments = X.shape[2]
        self.points_per_patch = X.shape[3]
        self.n_subjects = len(np.unique(self.subjects))
        
        print(f"Regression Dataset created:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Channels: {self.n_channels}")
        print(f"  Time segments: {self.time_segments}")
        print(f"  Points per patch: {self.points_per_patch}")
        print(f"  Unique subjects: {self.n_subjects}")
        print(f"  Target range: [{np.min(y):.2f}, {np.max(y):.2f}]")
        print(f"  Target mean: {np.mean(y):.2f} ± {np.std(y):.2f}")
    
    def __len__(self):
        """Return the total number of samples"""
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (eeg_data, target, subject_id)
                - eeg_data: Tensor of shape (n_channels, time_segments, points_per_patch)
                - target: Scalar tensor (regression target)
                - subject_id: Subject identifier
        """
        sample = self.X[idx]
        target = self.y[idx]
        subject_id = self.subjects[idx]
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target, subject_id
    
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
            'n_subjects': self.n_subjects,
            'input_shape': (self.n_channels, self.time_segments, self.points_per_patch),
            'target_stats': {
                'min': float(torch.min(self.y)),
                'max': float(torch.max(self.y)),
                'mean': float(torch.mean(self.y)),
                'std': float(torch.std(self.y))
            },
            'samples_per_subject': {
                int(subj): int((self.subjects == subj).sum())
                for subj in np.unique(self.subjects)
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
    print("Testing EEGRegressionDataset...")
    
    # Create dummy data
    n_samples = 100
    n_channels = 22
    time_segments = 4
    points_per_patch = 200
    n_subjects = 10
    
    X_dummy = np.random.randn(n_samples, n_channels, time_segments, points_per_patch)
    y_dummy = np.random.uniform(10, 50, n_samples)  # Raven scores range 10-50
    subjects_dummy = np.repeat(np.arange(n_subjects), n_samples // n_subjects)
    
    # Create dataset
    dataset = EEGRegressionDataset(X_dummy, y_dummy, subjects_dummy)
    
    # Test dataset access
    print(f"\nTesting dataset access...")
    sample, target, subject = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Target: {target.item():.2f}")
    print(f"Subject ID: {subject}")
    
    # Get data info
    print(f"\nDataset info:")
    info = dataset.get_data_info()
    for key, value in info.items():
        if key != 'samples_per_subject':
            print(f"  {key}: {value}")
    
    # Test with augmentation
    print(f"\nTesting with augmentation...")
    transform = torch.nn.Sequential(
        RandomTimeShift(max_shift=5),
        RandomAmplitudeScale(scale_range=(0.95, 1.05)),
        AddGaussianNoise(std=0.005)
    )
    dataset_aug = EEGRegressionDataset(X_dummy, y_dummy, subjects_dummy, transform=transform)
    sample_aug, _, _ = dataset_aug[0]
    print(f"Augmented sample shape: {sample_aug.shape}")
    
    print("\n✓ All tests passed!")
