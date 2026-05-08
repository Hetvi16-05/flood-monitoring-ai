import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import sys
import cv2

# Add project src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.segmentation_v2 import create_deeplabv3plus as FloodNet
from models.flood_transformer import SwinFloodNet
from models.segformer_model import SegFormerFlood
from models.custom_dataset import FloodCustomDataset
from training.losses import DiceFocalLoss

def calculate_metrics(pred, target, threshold=0.3):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    p = pred[:, 1].cpu().numpy().flatten()
    t = (target == 1).cpu().numpy().flatten()
    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    iou = intersection / (union + 1e-6)
    dice = (2. * intersection) / (p.sum() + t.sum() + 1e-6)
    precision = intersection / (p.sum() + 1e-6)
    recall = intersection / (t.sum() + 1e-6)
    return iou, dice, precision, recall

def save_visual_check(epoch, data, mask, output, save_dir, threshold=0.3):
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    img = (data[0, :3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gt = cv2.cvtColor((mask[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pred = (torch.sigmoid(output[0, 1]).cpu().numpy() > threshold).astype(np.uint8) * 255
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    gt[mask[0].cpu().numpy() == 1] = [0, 255, 0]
    pred[pred[:,:,0] == 255] = [0, 0, 255]
    cv2.imwrite(str(save_dir / f"epoch_{epoch+1}_v3_1.png"), np.hstack([img, gt, pred]))

def start_training():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATASET_ROOT = PROJECT_ROOT / "dataset_split"
    WEIGHTS_DIR = PROJECT_ROOT / "weights"; WEIGHTS_DIR.mkdir(exist_ok=True)
    VISUAL_DIR = PROJECT_ROOT / "visual_val_v3_1"
    
    NUM_CLASSES = 2
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
    LEARNING_RATE = 1e-4
    EPOCHS = int(os.getenv("EPOCHS", 40))
    USE_AMP = os.getenv("USE_AMP", "0") == "1"
    
    train_dataset = FloodCustomDataset(str(DATASET_ROOT / "train" / "images"), 
                                       masks_dir=str(DATASET_ROOT / "train" / "masks_hq"), balance=True)
    val_dataset = FloodCustomDataset(str(DATASET_ROOT / "val" / "images"), 
                                     masks_dir=str(DATASET_ROOT / "val" / "masks_hq"), balance=False)
    
    # Optimization for Mac: persistent_workers=True
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)

    model_type = os.getenv("MODEL_TYPE", "segformer")
    if model_type == "segformer":
        model = SegFormerFlood(num_classes=NUM_CLASSES, in_channels=6).to(device)
        weights_name = "segformer_flood_v3_1.pth"
    elif model_type == "swin_transformer":
        model = SwinFloodNet(num_classes=NUM_CLASSES, in_channels=6).to(device)
        weights_name = "swin_flood_net_v3_1.pth"
    else:
        model = FloodNet(num_classes=NUM_CLASSES).to(device)
        weights_name = "flood_net_v3_1.pth"
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # Scheduler: Break plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = DiceFocalLoss()
    
    # Gradient Scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    best_iou = 0.0
    for epoch in range(EPOCHS):
        model.train(); train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for data, mask, _ in pbar:
            data, mask = data.to(device), mask.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                output = model(data)
                loss = criterion(output, mask)
            
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval(); v_metrics = {"loss": 0, "iou": 0, "dice": 0, "prec": 0, "rec": 0}
        with torch.no_grad():
            for i, (data, mask, _) in enumerate(val_loader):
                data, mask = data.to(device), mask.to(device)
                output = model(data); v_metrics["loss"] += criterion(output, mask).item()
                iou, dice, prec, rec = calculate_metrics(output, mask)
                v_metrics["iou"] += iou; v_metrics["dice"] += dice; v_metrics["prec"] += prec; v_metrics["rec"] += rec
                if i == 0: save_visual_check(epoch, data, mask, output, VISUAL_DIR)
        
        num_v = len(val_loader); avg_iou = v_metrics['iou']/num_v
        print(f"📊 Epoch {epoch+1} Summary: IoU: {avg_iou:.4f} | Prec: {v_metrics['prec']/num_v:.4f} | Rec: {v_metrics['rec']/num_v:.4f}")
        
        # Scheduler update on IoU
        scheduler.step(avg_iou)
        
        # SAVE BEST ON IOU
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), str(WEIGHTS_DIR / weights_name))
            print(f"⭐ New Best Model (IoU: {avg_iou:.4f})")

if __name__ == "__main__":
    start_training()
