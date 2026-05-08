import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import sys
import cv2
import random
import torch.nn.functional as F

# Professional path management
sys.path.append(str(Path(__file__).parent.parent))

from models.segformer_model import SegFormerFlood
from models.custom_dataset import FloodCustomDataset

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred_softmax = torch.softmax(pred, dim=1)
        pred_fg = pred_softmax[:, 1]
        target_fg = (target == 1).float()
        
        bce = F.binary_cross_entropy(pred_fg, target_fg, reduction='none')
        p_t = pred_fg * target_fg + (1 - pred_fg) * (1 - target_fg)
        focal_loss = self.alpha * (1 - p_t)**self.gamma * bce
        
        intersection = (pred_fg * target_fg).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_fg.sum() + target_fg.sum() + self.smooth)
        
        return focal_loss.mean() + dice_loss

def start_elite_training():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ ELITE SPRINT: Targeting Speed + Precision on {device}")
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATASET_ROOT = PROJECT_ROOT / "dataset_split"
    WEIGHTS_DIR = PROJECT_ROOT / "weights"; WEIGHTS_DIR.mkdir(exist_ok=True)
    
    # 1. ELITE HYPERPARAMETERS
    BATCH_SIZE = 16 
    EPOCHS = 30
    LR_MAX = 8e-4
    IMAGES_PER_EPOCH = 2000 
    
    # 2. DATA LOADERS
    full_train_dataset = FloodCustomDataset(str(DATASET_ROOT / "train" / "images"), 
                                            masks_dir=str(DATASET_ROOT / "train" / "masks_hq"), 
                                            balance=True, augment=True)
    val_dataset = FloodCustomDataset(str(DATASET_ROOT / "val" / "images"), 
                                     masks_dir=str(DATASET_ROOT / "val" / "masks_hq"), 
                                     balance=False, augment=False)

    # 3. HIGH-CAPACITY MODEL (SegFormer-B0)
    model = SegFormerFlood(num_classes=2, in_channels=6).to(device)
    
    # [RESUME MECHANISM]
    RESUME_PATH = WEIGHTS_DIR / "rainwise_v3_1_expert.pth"
    best_iou = 0.0
    if RESUME_PATH.exists():
        print(f"♻️ RESUMING: Loading existing weights from {RESUME_PATH.name}")
        try:
            model.load_state_dict(torch.load(RESUME_PATH, map_location=device))
            print("✅ Weights Loaded Successfully.")
        except Exception as e:
            print(f"⚠️ Could not resume: {e}.")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX/10, weight_decay=0.01)
    
    steps_per_epoch = IMAGES_PER_EPOCH // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, steps_per_epoch=steps_per_epoch, epochs=EPOCHS
    )
    
    criterion = DiceFocalLoss()
    
    for epoch in range(EPOCHS):
        indices = random.sample(range(len(full_train_dataset)), IMAGES_PER_EPOCH)
        train_subset = Subset(full_train_dataset, indices)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        
        model.train()
        pbar = tqdm(train_loader, desc=f"Elite Epoch {epoch+1}/{EPOCHS}")
        for data, mask, _ in pbar:
            data, mask = data.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Fast Validation
        model.eval(); total_iou = 0
        val_subset_indices = random.sample(range(len(val_dataset)), min(500, len(val_dataset)))
        val_subset = Subset(val_dataset, val_subset_indices)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for data, mask, _ in val_loader:
                data, mask = data.to(device), mask.to(device)
                output = model(data)
                pred = torch.softmax(output, dim=1)[:, 1] > 0.5
                t = (mask == 1)
                inter = (pred & t).sum().item()
                union = (pred | t).sum().item()
                total_iou += inter / (union + 1e-6)
        
        avg_iou = total_iou / len(val_loader)
        print(f"📊 Accuracy: {avg_iou:.4f}")
        
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), str(WEIGHTS_DIR / "rainwise_v3_1_expert.pth"))
            print(f"⭐ [ELITE SAVED] IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    start_elite_training()
