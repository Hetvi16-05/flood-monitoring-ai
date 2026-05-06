import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import random

class FloodCustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, labels_dir=None, img_size=(256, 256), balance=True, augment=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.labels_dir = Path(labels_dir) if labels_dir else (self.images_dir.parent / "labels")
        self.img_size = img_size
        self.balance = balance
        self.augment = augment
        
        all_samples = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.png'))])
        self.pos_samples = []
        self.neg_samples = []
        for f in all_samples:
            mask_p = self.masks_dir / f"{Path(f).stem}.png" if self.masks_dir else None
            if mask_p and mask_p.exists():
                self.pos_samples.append(f)
            else:
                self.neg_samples.append(f)
        
        if self.balance and len(self.pos_samples) > 0:
            target_pos_count = len(self.neg_samples) // 2
            self.active_samples = self.pos_samples * (target_pos_count // len(self.pos_samples))
            self.active_samples += self.neg_samples
            random.shuffle(self.active_samples)
        else:
            self.active_samples = all_samples

    def apply_monsoon_augmentations(self, frame, mask):
        # 1. Geometric: Shift, Scale, Rotate
        if random.random() > 0.5:
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), random.uniform(-15, 15), random.uniform(0.8, 1.2))
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 2. Environmental: Simulated Rain/Noise
        if random.random() > 0.3:
            noise = np.random.normal(0, random.randint(5, 15), frame.shape).astype(np.int16)
            frame = cv2.add(frame.astype(np.int16), noise)
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # 3. Weather: Simulated Fog/Blur
        if random.random() > 0.3:
            k = random.choice([3, 5, 7])
            frame = cv2.GaussianBlur(frame, (k, k), 0)

        # 4. Flips
        if random.random() > 0.5: frame = cv2.flip(frame, 1); mask = cv2.flip(mask, 1)
        
        return frame, mask

    def __len__(self):
        return len(self.active_samples)

    def __getitem__(self, idx):
        img_name = self.active_samples[idx]
        base_name = os.path.splitext(img_name)[0]
        img_path = str(self.images_dir / img_name)
        frame = cv2.imread(img_path)
        if frame is None: return torch.zeros((6, *self.img_size)), torch.zeros(self.img_size, dtype=torch.long), []
        
        frame = cv2.resize(frame, self.img_size)
        mask = np.zeros(self.img_size, dtype=np.uint8)
        if self.masks_dir:
            mask_path = self.masks_dir / f"{base_name}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            
        if self.augment:
            frame, mask = self.apply_monsoon_augmentations(frame, mask)
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        g, r = rgb[:,:,1], rgb[:,:,0]
        ndwi = ((g - r) / (g + r + 1e-6) + 1) / 2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        texture = cv2.normalize(np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)**2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)**2), None, 0, 1, cv2.NORM_MINMAX)
        h, w = self.img_size
        elevation = np.linspace(1, 0, h).reshape(h, 1).repeat(w, axis=1).astype(np.float32)
        
        six_channel = np.zeros((h, w, 6), dtype=np.float32)
        six_channel[:, :, :3] = rgb
        six_channel[:, :, 3] = ndwi
        six_channel[:, :, 4] = texture
        six_channel[:, :, 5] = elevation
        
        return torch.from_numpy(six_channel).permute(2, 0, 1).float(), torch.from_numpy(mask).long(), []
