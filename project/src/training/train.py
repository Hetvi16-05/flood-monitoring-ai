import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from models.dataset import SegmentationDataset
from models.segmentation_v2 import create_deeplabv3plus
from preprocessing.augment import get_production_augmentations
from config import (
    EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    WEIGHTS_DIR,
    LOGS_DIR,
    SPLIT_DIR
)

# Optimized Batch Size for 5-Channel DeepLabV3+ on Apple MPS Memory
BATCH_SIZE = 8

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target):
        smooth = 1e-6
        valid_mask = (target != 255).unsqueeze(1).float()
        inputs = F.softmax(inputs, dim=1)
        target_fixed = target.clone()
        target_fixed[target == 255] = 0
        target_one_hot = F.one_hot(target_fixed, self.n_classes).permute(0, 3, 1, 2).float()
        
        inputs = inputs * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        intersection = torch.sum(inputs * target_one_hot, (2, 3))
        union = torch.sum(inputs + target_one_hot, (2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def compute_iou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    valid_mask = (target != 255)
    iou = 0
    count = 0
    
    for cls in range(num_classes):
        p = (pred == cls) & valid_mask
        t = (target == cls) & valid_mask
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        
        if union > 0:
            iou += inter / union
            count += 1
            
    return iou / count if count > 0 else torch.tensor(0.0)

def train_model(epochs_override=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Initializing Optimized Retraining Pipeline with EarlyStopping on {device}")
    
    global EPOCHS
    if epochs_override is not None:
        EPOCHS = epochs_override

    # Datasets
    train_img_dir = SPLIT_DIR / "train" / "images"
    train_mask_dir = SPLIT_DIR / "train" / "masks"
    val_img_dir = SPLIT_DIR / "val" / "images"
    val_mask_dir = SPLIT_DIR / "val" / "masks"

    # Albumentations applied strictly to training
    train_dataset = SegmentationDataset(
        img_dir=str(train_img_dir),
        mask_dir=str(train_mask_dir),
        transform=get_production_augmentations()
    )
    
    val_dataset = SegmentationDataset(
        img_dir=str(val_img_dir),
        mask_dir=str(val_mask_dir),
        transform=None
    )

    # For MacOS/MPS Stability, num_workers=0 and pin_memory=False is mandatory
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # Model Initialization (5-Channel Setup)
    model = create_deeplabv3plus(in_channels=5, num_classes=NUM_CLASSES)
    
    # Checkpoint Resume Logic
    v2_weights_path = os.path.join(WEIGHTS_DIR, "best_model_v2.pth")
    if os.path.exists(v2_weights_path):
        print(f"🔄 Resuming from cached V2 checkpoint: {v2_weights_path}")
        try:
            model.load_state_dict(torch.load(v2_weights_path, map_location=device))
        except Exception as e:
            print(f"⚠️ Warning during strict weight reload: {e}")
            
    model.to(device)

    # -----------------------------
    # ADVANCED LOSS CONFIGURATION
    # -----------------------------
    class_weights = torch.tensor([1.5, 0.8, 1.2, 0.8, 2.5, 3.0], dtype=torch.float32).to(device)
    
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=255)
    dice_loss = DiceLoss(NUM_CLASSES)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # -----------------------------
    # OneCycleLR SCHEDULER (Clamped)
    # -----------------------------
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4, # Firm max LR clamp to strictly prevent catastrophic forgetting
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS
    )

    best_iou = 0.0 
    epochs_no_improve = 0
    patience = 3 # EarlyStopping patience

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_file = LOGS_DIR / "training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_IoU", "LR"])

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (images, masks) in enumerate(progress_bar):
            # Optimizations for MPS Backend
            images = images.to(device).float()
            masks = masks.to(device).long()
            
            optimizer.zero_grad()
            out = model(images)
            
            # True Multi-Objective Criterion combining localized gradients with spatial intersections
            loss = 0.5 * focal_loss(out, masks) + 0.5 * dice_loss(out, masks)
            loss.backward()
            
            optimizer.step()
            scheduler.step() # OneCycleLR ticks intrinsically inside the batch loop
            
            train_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            if epochs_override == 1 and i >= 10: # Stop early on dry-run
                 break
                 
        train_loss /= (i + 1)

        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images = images.to(device).float()
                masks = masks.to(device).long()
                
                out = model(images)
                loss = 0.5 * focal_loss(out, masks) + 0.5 * dice_loss(out, masks)
                
                val_loss += loss.item()
                val_iou += compute_iou(out, masks, NUM_CLASSES).item()
                if epochs_override == 1 and i >= 3: # Stop early on dry-run
                     break
                     
        val_loss /= (i + 1)
        val_iou /= (i + 1)
        
        current_lr = scheduler.get_last_lr()[0]

        print(f"📊 Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f} | IoU {val_iou:.4f} | LR: {current_lr:.6f}")
        
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss, val_iou, current_lr])

        # Overhauled checkpointing logic with EarlyStopping
        if val_iou > best_iou:
            best_iou = val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_model_v2.pth"))
            print("✅ Best IoU detected. V2 Weights Accurately Preserved (best_model_v2.pth)")
        else:
            epochs_no_improve += 1
            print(f"⚠️ Validation IoU did not improve. EarlyStopping Counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered at Epoch {epoch+1}. Model has converged safely.")
                break

if __name__ == "__main__":
    import sys
    epochs_override = None
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        epochs_override = 1
        
    train_model(epochs_override)