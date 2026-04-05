import os
import cv2
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from config import CLEAN_DIR, IMG_SIZE, PROJECT_ROOT
from preprocessing.feature_engineering import extract_hybrid_features
from preprocessing.augment import apply_augmentations

class SegmentationDataset(Dataset):
    def __init__(self, img_dir=None, mask_dir=None, transform=None):
        self.img_dir = img_dir if img_dir else os.path.join(CLEAN_DIR, "images")
        self.mask_dir = mask_dir if mask_dir else os.path.join(CLEAN_DIR, "masks")
        self.transform = transform # Can be Albumentations or Torchvision
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 1. LOAD IMAGE & MASK
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        from config import CLASSES
        class_map = {cls: i for i, cls in enumerate(CLASSES)}
        label = 0
        for cls, i in class_map.items():
            if img_name.startswith(cls):
                label = i
                break

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.full((img.shape[0], img.shape[1]), label, dtype=np.uint8)

        # 2. APPLY AUGMENTATIONS (ALBUMENTATIONS)
        if self.transform:
            img, mask = apply_augmentations(img, mask, self.transform)
        else:
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        # 3. FEATURE ENGINEERING (spatial + texture)
        # We extract features AFTER augmentations or ON THE FLY to ensure 5-channel consistency
        hybrid_features = extract_hybrid_features(img) # Extracts Canny and LBP
        
        # Merge: RGB (3) + Hybrid (2) = 5 Channels
        # Normalization for Image: 0-255 -> 0-1
        img_norm = img.astype(np.float32) / 255.0
        
        # Final Shape: (H, W, 5)
        combined = np.concatenate([img_norm, hybrid_features], axis=-1)
        
        # 4. CHANNELS FIRST (C, H, W)
        combined_tensor = torch.from_numpy(combined).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()

        return combined_tensor, mask_tensor