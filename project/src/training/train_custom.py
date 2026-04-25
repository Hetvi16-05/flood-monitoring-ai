import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Add project/src to sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.flood_net import FloodNet
from models.emergency_detector import EmergencyDetector
from models.custom_dataset import FloodCustomDataset
from training.losses import WaterFocalDiceLoss, FloodDetectionLoss
from preprocessing.flood_augmentations import apply_flood_augmentation_pipeline

# Training Hyperparameters
BATCH_SIZE = 4 # Reduced for heavy 6-channel and 3D models
LEARNING_RATE = 1e-4
EPOCHS = 10

class CustomTrainingPipeline:
    def __init__(self, model_type='flood_net', device='cpu'):
        self.device = device
        self.model_type = model_type
        
        if model_type == 'flood_net':
            # 8 classes, 6-channel input
            self.model = FloodNet(num_classes=8).to(device)
            self.criterion = WaterFocalDiceLoss()
        elif model_type == 'emergency':
            # 5 classes
            self.model = EmergencyDetector(num_classes=5).to(device)
            self.criterion = FloodDetectionLoss()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        progress = tqdm(dataloader, desc=f"Training {self.model_type}")
        for i, (data, masks, labels) in enumerate(progress):
            # Move to device
            data = data.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                outputs = self.model(data)
                loss = self.criterion(outputs, masks)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return total_loss / len(dataloader)

def start_training():
    # 1. Hardware Selection (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon GPU (MPS) for training!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using NVIDIA GPU (CUDA) for training!")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU not found, training on CPU. This will be slow.")

    # 2. Paths
    ROOT = Path("/Users/HetviSheth/Flood_Prediction")
    DATASET_ROOT = ROOT / "project" / "dataset_split"
    
    TRAIN_IMAGES = DATASET_ROOT / "train" / "images"
    TRAIN_MASKS = DATASET_ROOT / "train" / "masks"
    VAL_IMAGES = DATASET_ROOT / "val" / "images"
    VAL_MASKS = DATASET_ROOT / "val" / "masks"
    
    # 3. Model & Hyperparameters
    num_classes = 8
    batch_size = 16 # Adjust based on RAM
    epochs = 10
    learning_rate = 1e-4

    # 4. Loaders
    train_dataset = FloodCustomDataset(str(TRAIN_IMAGES), str(TRAIN_MASKS), img_size=(256, 256))
    val_dataset = FloodCustomDataset(str(VAL_IMAGES), str(VAL_MASKS), img_size=(256, 256))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = FloodNet(num_classes=num_classes).to(device)
    
    # 5. Loss & Optimizer
    criterion = WaterFocalDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 6. Training Loop with Early Stopping
    print(f"🔥 Starting Custom Training Loop ({epochs} epochs)...")
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        for batch_idx, (data, mask, _) in enumerate(train_loader):
            data, mask = data.to(device), mask.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, mask)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Training | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, mask, _ in val_loader:
                data, mask = data.to(device), mask.to(device)
                output = model(data)
                loss = criterion(output, mask)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"✅ Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- Early Stopping & Best Model Save ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            weights_path = ROOT / "weights" / "flood_net_custom.pth"
            weights_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), str(weights_path))
            print(f"⭐ New Best Model Saved (Val Loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement in Val Loss. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early stopping triggered. Training terminated.")
                break

if __name__ == "__main__":
    start_training()
