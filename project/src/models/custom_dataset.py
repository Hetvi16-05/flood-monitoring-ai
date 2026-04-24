import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class FloodCustomDataset(Dataset):
    """
    Custom Dataset for 6-channel FloodNet training.
    Loads .npy files (6-channel) and .png masks (8-class).
    """
    def __init__(self, data_dir, mask_dir, labels_dir=None, transform=None):
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.transform = transform
        
        self.samples = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        base_name = os.path.splitext(name)[0]
        
        # 1. Load 6-channel data
        data = np.load(self.data_dir / name) # [H, W, 6]
        
        # 2. Load 8-class mask
        mask_path = self.mask_dir / f"{base_name}.png"
        if mask_path.exists():
            import cv2
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (data.shape[1], data.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            # Fallback if no mask exists (dummy)
            mask = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)
            
        # 3. Load Detection Labels (if any)
        # Format: [class, x, y, w, h, submersion_conf]
        labels = []
        if self.labels_dir:
            label_path = self.labels_dir / f"{base_name}.txt"
            if label_path.exists():
                labels = np.loadtxt(str(label_path)).reshape(-1, 6)
        
        # Convert to Tensor
        data_tensor = torch.from_numpy(data).permute(2, 0, 1).float() # [6, H, W]
        mask_tensor = torch.from_numpy(mask).long() # [H, W]
        
        return data_tensor, mask_tensor, labels
