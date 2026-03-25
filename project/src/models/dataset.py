import os
import cv2
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

# -----------------------------
# PATH FIX
# -----------------------------
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import CLEAN_DIR, IMG_SIZE


class SegmentationDataset(Dataset):

    def __init__(self, transform=None):

        self.img_dir = os.path.join(CLEAN_DIR, "images")
        self.mask_dir = os.path.join(CLEAN_DIR, "masks")

        self.transform = transform

        self.images = sorted(
            [
                f for f in os.listdir(self.img_dir)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(
            self.img_dir,
            img_name
        )

        mask_name = (
            img_name
            .replace(".jpg", ".png")
            .replace(".jpeg", ".png")
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        # -----------------------------
        # IMAGE
        # -----------------------------
        img = cv2.imread(img_path)
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # -----------------------------
        # MASK
        # -----------------------------
        class_map = {
            "water": 0,
            "road": 1,
            "building": 2,
            "vegetation": 3
        }

        # Determine label from filename prefix (fallback or for mapping)
        label = 0
        for cls, i in class_map.items():
            if img_name.startswith(cls + "_"):
                label = i
                break

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.int64)
            
            unique = np.unique(mask)
            
            # Robust mapping for background ignore:
            # We want only the primary object (label) to have its index.
            # Everything else becomes 255 (ignore).
            
            new_mask = np.full(mask.shape, 255, dtype=np.int64)
            
            # If the mask is legacy 0/255, then 255 is the object.
            # If it's already converted, then 'label' is the object.
            if 255 in unique and label not in unique:
                 new_mask[mask == 255] = label
            else:
                 new_mask[mask == label] = label
            
            mask = new_mask

        else:
            mask = np.full(
                (IMG_SIZE[1], IMG_SIZE[0]),
                label,
                dtype=np.uint8
            )

        # -----------------------------
        # RESIZE (ONLY HERE)
        # -----------------------------
        img = cv2.resize(
            img,
            IMG_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            IMG_SIZE,
            interpolation=cv2.INTER_NEAREST
        )

        # -----------------------------
        # NORMALIZE IMAGE
        # -----------------------------
        img = img.astype(np.float32) / 255.0

        # -----------------------------
        # TO TENSOR
        # -----------------------------
        img = torch.from_numpy(
            img
        ).permute(2, 0, 1).float()

        mask = torch.from_numpy(
            mask
        ).long()

        # -----------------------------
        # TRANSFORM (ONLY NORMALIZE)
        # -----------------------------
        if self.transform:
            img = self.transform(img)

        return img, mask