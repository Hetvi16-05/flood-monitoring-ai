import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path
import numpy as np

# Add project/src to sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.flood_net import FloodNet
from models.flood_transformer import SwinFloodNet
from models.emergency_detector import EmergencyDetector
from models.custom_dataset import FloodCustomDataset
from training.losses import WaterFocalDiceLoss, FloodDetectionLoss
from preprocessing.flood_augmentations import apply_flood_augmentation_pipeline

# --- METRIC TRACKING ---
def calculate_metrics(pred, target, num_classes=8):
    """
    Calculate IoU and Dice Score for each class.
    """
    pred = torch.argmax(pred, dim=1)
    ious = []
    dices = []
    
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        total = (pred == cls).sum().item() + (target == cls).sum().item()
        
        iou = inter / (union + 1e-6)
        dice = (2 * inter) / (total + 1e-6)
        
        ious.append(iou)
        dices.append(dice)
        
    return np.mean(ious), np.mean(dices)

# --- ENHANCED TRAINING PIPELINE ---
def start_training():
    # 1. Hardware Selection
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🚀 Device selected: {device}")

    # 2. Paths
    ROOT = Path("/Users/HetviSheth/Flood_Prediction")
    PROJECT_ROOT = ROOT / "project"
    DATASET_ROOT = PROJECT_ROOT / "dataset_split"
    WEIGHTS_DIR = PROJECT_ROOT / "weights"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    
    # 3. Hyperparameters
    NUM_CLASSES = 8
    BATCH_SIZE = 2 # Reduced for small dataset test
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    
    # 4. Loaders (Using the newly split 14,000+ images)
    train_dataset = FloodCustomDataset(str(DATASET_ROOT / "train" / "images"), 
                                       img_size=(256, 256))
    val_dataset = FloodCustomDataset(str(DATASET_ROOT / "val" / "images"), 
                                     img_size=(256, 256))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 5. Model Initialization (Using Pretrained Swin-V2)
    model_type = os.environ.get("MODEL_TYPE", "swin_transformer")
    if model_type == "swin_transformer":
        model = SwinFloodNet(num_classes=NUM_CLASSES, in_channels=6).to(device)
        weights_name = "swin_flood_net.pth"
    else:
        model = FloodNet(num_classes=NUM_CLASSES).to(device)
        weights_name = "flood_net_custom.pth"
    
    # 6. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = WaterFocalDiceLoss()
    
    # --- RESUME LOGIC ---
    last_checkpoint_path = WEIGHTS_DIR / f"last_{weights_name}"
    start_epoch = 0
    best_val_loss = float('inf')

    if last_checkpoint_path.exists():
        print(f"🔄 Found checkpoint at {last_checkpoint_path}. Resuming...")
        checkpoint = torch.load(str(last_checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"👉 Resuming from Epoch {start_epoch + 1}")

    # 7. Training Loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for data, mask, _ in pbar:
            data, mask = data.to(device), mask.to(device)
            
            # Apply strong flood augmentations (Custom Dataset + Manual Flip)
            if np.random.rand() > 0.5:
                data = torch.flip(data, dims=[3])
                mask = torch.flip(mask, dims=[2])
                
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, mask)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation & Metrics ---
        model.eval()
        val_loss = 0
        total_iou = 0
        total_dice = 0
        
        with torch.no_grad():
            for data, mask, _ in val_loader:
                data, mask = data.to(device), mask.to(device)
                output = model(data)
                val_loss += criterion(output, mask).item()
                
                iou, dice = calculate_metrics(output, mask, num_classes=NUM_CLASSES)
                total_iou += iou
                total_dice += dice
        
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        print(f"📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # --- Saving & Checkpointing ---
        # 1. Save Last (Checkpoint)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, str(last_checkpoint_path))
        
        # 2. Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            weights_path = WEIGHTS_DIR / weights_name
            torch.save(model.state_dict(), str(weights_path))
            print(f"⭐ New Best Model Saved (Val Loss: {avg_val_loss:.4f})")

if __name__ == "__main__":
    start_training()
