import os
from math import ceil
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


class GroupedActivationDataset(Dataset):
    def __init__(self, root_dir, group_size):
        self.file_paths = sorted([
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) if f.endswith((".npy", ".npy.npz"))
        ])
        self.group_size = group_size
        self.num_groups = ceil(len(self.file_paths) / group_size)

    def __len__(self):
        return self.num_groups
    
    def __getitem__(self, idx):
        start = idx * self.group_size
        end = min(start + self.group_size, len(self.file_paths))
        group_files = self.file_paths[start:end]

        group_data = []
        for fp in group_files:
            if fp.endswith('.npy'):
                arr = np.load(fp, mmap_mode='r')
            elif fp.endswith('.npy.npz'):
                with np.load(fp) as data:
                    arr = data[data.files[0]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                group_data.append(torch.from_numpy(arr).float())
        
        batch = torch.cat(group_data, dim=0)
        return batch