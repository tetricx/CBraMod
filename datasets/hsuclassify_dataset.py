import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        super(CustomDataset, self).__init__()
        
        if mode == 'train':
            self.X = np.load(os.path.join(data_dir, 'X_train.npy'))
            self.y = np.load(os.path.join(data_dir, 'y_train.npy'))
        elif mode == 'val':
            self.X = np.load(os.path.join(data_dir, 'X_val.npy'))
            self.y = np.load(os.path.join(data_dir, 'y_val.npy'))
        elif mode == 'test':
            self.X = np.load(os.path.join(data_dir, 'X_test.npy'))
            self.y = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        data = self.X[idx]
        label = self.y[idx]
        return data/100, label
    
    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
    
    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        
        print(f"Dataset sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        print(f"Total: {len(train_set) + len(val_set) + len(test_set)}")
        
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=True,
            ),
        }
        return data_loader