import os
from ultralytics import YOLO

def train_rainwise_yolo():
    """
    Launches YOLOv8 segmentation training for flood detection.
    """
    # 1. Load a pre-trained model (YOLOv8n-seg is best for real-time edge devices)
    model = YOLO('yolov8n-seg.pt')

    # 2. Train the model
    # data: path to data.yaml
    # epochs: set higher for production (e.g., 100)
    # imgsz: standard YOLO size
    # device: mps for Apple Silicon
    print("🚀 Starting YOLOv8 Training for RAINWISE...")
    results = model.train(
        data='project/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='mps',
        project='project/runs',
        name='flood_segmentation',
        augment=True # Enable built-in augmentations (flips, mosaics, etc.)
    )

    print("✅ Training complete. Results saved in project/runs/flood_segmentation")

if __name__ == "__main__":
    train_rainwise_yolo()
