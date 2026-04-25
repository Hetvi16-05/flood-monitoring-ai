import os
import torch
from ultralytics import YOLO
from pathlib import Path

def train_yolo(data_yaml='project/data.yaml', epochs=50, batch=16, imgsz=640):
    """
    Launches YOLOv8 Training for RAINWISE V3.
    Supports Apple Silicon MPS.
    """
    # 1. Select Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🚀 Device selected: {device}")
    
    # 2. Initialize Model (Nano version for speed and real-time efficiency)
    model = YOLO('yolov8n.pt')
    
    # 3. Start Training
    print(f"🔥 Starting Training: {epochs} epochs, batch {batch}, size {imgsz}...")
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project='project/runs',
        name='rainwise_v3',
        exist_ok=True,
        augment=True
    )
    
    print(f"✅ Training Complete. Best model saved in project/runs/rainwise_v3/weights/best.pt")

if __name__ == "__main__":
    train_yolo()