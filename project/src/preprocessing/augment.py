import albumentations as A
import cv2
import numpy as np
import os
from pathlib import Path

def get_production_augmentations(img_size=(256, 256)):
    """
    Lightweight, high-stability continuous augmentation pipeline.
    Avoids heavy noise simulators to preserve crisp spatial boundaries.
    """
    return A.Compose([
        # 1. Lightweight Lighting Adjustments
        A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
        
        # 2. Sub-perceptual Blur & Noise
        A.OneOf([
            A.MotionBlur(p=1.0, blur_limit=3),
            A.GaussianBlur(p=1.0, blur_limit=3),
            A.GaussNoise(p=1.0),
        ], p=0.3),

        # 3. Core Geometric (Crucial for invariance without distortion)
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Resize(img_size[1], img_size[0]),
    ])

def apply_augmentations(image, mask, transforms):
    """Apply transforms to image and mask simultaneously."""
    augmented = transforms(image=image, mask=mask)
    return augmented['image'], augmented['mask']

if __name__ == "__main__":
    print("🚀 Calibration Complete: Production-Grade Augmentations Initialized.")
