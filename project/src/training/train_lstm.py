import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.flood_lstm import FloodLSTM
from config import LOGS_DIR, WEIGHTS_DIR, DEVICE

class FloodTimeSeriesDataset(Dataset):
    def __init__(self, csv_path, seq_length=12):
        self.seq_length = seq_length
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"DEBUG: Loaded CSV columns: {df.columns.tolist()}")
            
            # Map Risk levels to numerical values
            risk_mapping = {
                'LOW': 10, 'MEDIUM': 40, 'HIGH': 70, 'DANGEROUS': 90, 'EXTREME': 100
            }
            
            # Ensure Risk is string for mapping
            df['Risk_Level'] = df['Risk_Level'].astype(str).str.upper().str.strip()
            
            # Try to convert to numeric first, if fail, map from risk_mapping
            df['Risk_Score_Mapped'] = pd.to_numeric(df['Risk_Level'], errors='coerce')
            # Fill NaNs from mapping
            df.loc[df['Risk_Score_Mapped'].isna(), 'Risk_Score_Mapped'] = df.loc[df['Risk_Score_Mapped'].isna(), 'Risk_Level'].map(risk_mapping)
            # Fill remaining NaNs with 0
            df['Risk_Score_Mapped'] = df['Risk_Score_Mapped'].fillna(0)
            
            # Use Risk_Score if it exists in CSV
            if 'Risk_Score' in df.columns:
                df['Risk_Score_Final'] = pd.to_numeric(df['Risk_Score'], errors='coerce').fillna(df['Risk_Score_Mapped'])
            else:
                df['Risk_Score_Final'] = df['Risk_Score_Mapped']
            
            print(f"DEBUG: Unique Risk_Level values: {df['Risk_Level'].unique()}")
            print(f"DEBUG: Unique Risk_Score_Final values: {df['Risk_Score_Final'].unique()}")
            
            # Clean numerical columns
            for col in ['Water_P', 'Rain_mm']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else:
                    # Fallback for old names
                    old_col = 'Water %' if col == 'Water_P' else 'Rain'
                    df[col] = pd.to_numeric(df[old_col], errors='coerce').fillna(0)
                
            features = df[['Water_P', 'Rain_mm', 'Risk_Score_Final']].values.astype(np.float32)
        else:
            print(f"⚠️ Warning: {csv_path} not found. Generating synthetic data for demonstration.")
            features = self._generate_synthetic_data()
            
        # Normalization (simple min-max for demonstration)
        self.min_val = features.min(axis=0)
        self.max_val = features.max(axis=0) + 1e-6
        self.data = (features - self.min_val) / (self.max_val - self.min_val)
        
        self.sequences = []
        self.targets = []
        
        # Multi-step prediction: +1, +3, +6 steps ahead
        forecast_steps = [1, 3, 6]
        max_step = max(forecast_steps)
        
        for i in range(len(self.data) - seq_length - max_step):
            self.sequences.append(self.data[i : i + seq_length])
            
            # Risk score is at index 2
            t1 = self.data[i + seq_length, 2]
            t3 = self.data[i + seq_length + 2, 2]
            t6 = self.data[i + seq_length + 5, 2]
            
            self.targets.append([t1, t3, t6])
            
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def _generate_synthetic_data(self):
        # Generate 500 points of synthetic flood data
        t = np.linspace(0, 100, 500)
        rain = 20 * np.sin(t/5) + 30 + np.random.normal(0, 5, 500)
        water = 0.5 * rain + 10 * np.sin(t/10) + np.random.normal(0, 2, 500)
        risk = (water * 0.7 + rain * 0.3) + np.random.normal(0, 5, 500)
        
        # Clip to realistic ranges
        rain = np.clip(rain, 0, 100)
        water = np.clip(water, 0, 100)
        risk = np.clip(risk, 0, 100)
        
        return np.stack([water, rain, risk], axis=1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def train():
    # Paths
    csv_path = LOGS_DIR / "monitor_log.csv"
    save_path = WEIGHTS_DIR / "flood_lstm_v2.pth"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    
    # Hyperparameters
    SEQ_LENGTH = 12 # 12 steps of history
    BATCH_SIZE = 16
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    EPOCHS = 30
    
    # Dataset & Loader
    dataset = FloodTimeSeriesDataset(csv_path, seq_length=SEQ_LENGTH)
    if len(dataset) < BATCH_SIZE:
        print("❌ Dataset too small for training.")
        return
        
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model, Loss, Optimizer
    model = FloodLSTM(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"🚀 Starting LSTM training on {DEVICE}...")
    print(f"📊 Dataset size: {len(dataset)} sequences")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for seq, target in train_loader:
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}")
            
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'min_val': dataset.min_val,
        'max_val': dataset.max_val,
        'config': {
            'input_size': 3,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'seq_length': SEQ_LENGTH,
            'output_size': 3
        }
    }, save_path)
    
    print(f"✅ Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train()
