from ultralytics import YOLO
import torch
from pathlib import Path
import os

def train_segmentation_model():
    print("🚀 STARTING PIXEL-PERFECT MASTER TRAINING (YOLO12-Seg)")
    print("-------------------------------------------------------")
    print("💎 Goal: 100% Boundary Precision + Zero False Positives")
    
    # Paths
    YAML_PATH = "/Users/HetviSheth/Flood_Prediction/newmodel/master_data.yaml"
    SEG_CHECKPOINT = "/Users/HetviSheth/Flood_Prediction/newmodel/training_runs/rainwise_seg_master_final/weights/last.pt"
    
    # 1. Initialize or Smart-Resume
    if os.path.exists(SEG_CHECKPOINT):
        print(f"♻️ RESUMING: Loading master segmentation checkpoint from {SEG_CHECKPOINT}")
        model = YOLO(SEG_CHECKPOINT)
        resume_flag = True
    else:
        print("🆕 STARTING FRESH: Initializing Pixel-Perfect Segmentation Brain")
        try:
            model = YOLO("yolo12s-seg.pt")
            print("✅ Using Cutting-Edge YOLO12-Segmentation Weights")
        except:
            model = YOLO("yolo11s-seg.pt")
            print("✅ Using Hyper-Powerful YOLO11-Segmentation Weights")
        resume_flag = False
    
    # 2. HIGH-PRECISION TRAINING
    model.train(
        data=YAML_PATH,
        epochs=50,
        imgsz=640,
        batch=12, # Slightly smaller batch for segmentation memory
        device="mps", # Use Mac GPU
        patience=10, 
        save=True,
        project="newmodel/training_runs",
        name="rainwise_seg_master_final",
        resume=resume_flag,
        plots=True,
        overlap_mask=True, # Critical for complex flood shapes
        mask_ratio=1, # Perfect mask resolution
        lr0=0.01,
        warmup_epochs=3
    )
    
    print("\n✅ MASTER SEGMENTATION TRAINING COMPLETE!")
    print(f"⭐ Master weights saved in: newmodel/training_runs/rainwise_seg_master_final/weights/best.pt")

if __name__ == "__main__":
    train_segmentation_model()
