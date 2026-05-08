from ultralytics import YOLO
import torch
import os

def train_master_crocodile():
    print("🐊 RAINWISE - APEX PREDATOR MASTER TRAINING (YOLO12)")
    print("--------------------------------------------------")
    
    # 1. Device Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Using Device: {device.upper()}")
    
    # 2. Model Initialization (Detection)
    model_path = "yolo12n.pt" # Latest YOLO for high speed
    
    # Check for existing checkpoint to resume
    resume_path = "newmodel/crocodile_training/weights/last.pt"
    if os.path.exists(resume_path):
        print(f"🔄 Resuming from last checkpoint: {resume_path}")
        model = YOLO(resume_path)
    else:
        print(f"🆕 Starting fresh training with: {model_path}")
        model = YOLO(model_path)
    
    # 3. Training Loop
    model.train(
        data="newmodel/master_crocodile.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project="newmodel",
        name="crocodile_training",
        exist_ok=True,
        # Resume if last.pt exists
        resume=os.path.exists(resume_path)
    )
    
    print("\n✅ TRAINING COMPLETE!")
    print("Best weights saved in: newmodel/crocodile_training/weights/best.pt")

if __name__ == "__main__":
    train_master_crocodile()
