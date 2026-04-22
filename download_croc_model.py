"""
Download pre-trained crocodile detection model from Roboflow
"""
import os
import requests
from pathlib import Path

def download_crocodile_model():
    """
    Download a pre-trained YOLOv8 crocodile detection model
    """
    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    output_path = weights_dir / "crocodile_detector.pt"
    
    # Using a publicly available YOLOv8 model trained on crocodiles
    # This is a placeholder URL - in production, you would use a trained model
    # For now, we'll use the standard YOLOv8n as a base and note it needs training
    
    print("🐊 Downloading crocodile detection model...")
    
    # Since we don't have a direct download URL for a trained crocodile model,
    # we'll copy the standard YOLOv8n as a placeholder and train it later
    # In production, you would:
    # 1. Download from Roboflow Universe: https://universe.roboflow.com/
    # 2. Or train on your own dataset using: yolo detect train data=crocodile_dataset.yaml
    
    source_yolo = Path(__file__).parent / "yolov8n.pt"
    if source_yolo.exists():
        import shutil
        shutil.copy(source_yolo, output_path)
        print(f"✅ Copied base YOLOv8n to {output_path}")
        print("⚠️ Note: This is a base model. To detect crocodiles, you need to:")
        print("   1. Collect crocodile images")
        print("   2. Annotate them (use Roboflow or LabelImg)")
        print("   3. Train: yolo detect train data=crocodile.yaml model=crocodile_detector.pt epochs=50")
        return True
    else:
        print("❌ Base YOLOv8n not found. Please download yolov8n.pt first.")
        return False

if __name__ == "__main__":
    download_crocodile_model()
