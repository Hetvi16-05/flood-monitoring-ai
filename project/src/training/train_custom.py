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

def run_custom_training(model_type):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🔥 Starting Custom Training for {model_type} on {device}")
    
    # Paths
    ROOT = Path(__file__).resolve().parents[3] # Go up to Flood_Prediction/
    DATA_DIR = ROOT / "project" / "dataset_custom" / "processed_6c"
    MASK_DIR = ROOT / "project" / "dataset_custom" / "masks_8c"
    
    # Loader
    dataset = FloodCustomDataset(str(DATA_DIR), str(MASK_DIR))
    if len(dataset) == 0:
        print("❌ Dataset is empty. Run pre-processing first!")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    pipeline = CustomTrainingPipeline(model_type=model_type, device=device)
    
    for epoch in range(EPOCHS):
        loss = pipeline.train_epoch(dataloader)
        print(f"✅ Epoch {epoch+1}/{EPOCHS} complete. Loss: {loss:.4f}")
        
        # Save Weights
        save_path = ROOT / "project" / "weights" / f"custom_{model_type}.pth"
        torch.save(pipeline.model.state_dict(), str(save_path))

if __name__ == "__main__":
    import sys
    m_type = sys.argv[1] if len(sys.argv) > 1 else 'flood_net'
    run_custom_training(m_type)
