"""
Download pre-trained crocodile detection model from Roboflow Universe
"""
import requests
from pathlib import Path

def download_crocodile_model():
    """
    Download pre-trained crocodile detection model from Roboflow Universe
    Using direct URL download for a publicly available crocodile model
    """
    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    output_path = weights_dir / "crocodile_detector.pt"
    
    print("🐊 Downloading crocodile detection model...")
    
    # Try downloading from a public GitHub repository that has crocodile models
    # or use a known pre-trained model URL
    model_urls = [
        # Add direct download URLs here if available
        # For now, we'll provide manual instructions
    ]
    
    print("\n⚠️ Automatic download requires Roboflow API key.")
    print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 60)
    print("Option 1: Roboflow Universe (Recommended)")
    print("1. Visit: https://universe.roboflow.com/search?q=class:crocodile")
    print("2. Find a YOLOv8 crocodile model")
    print("3. Click 'Download' → 'Weights' → 'YOLOv8'")
    print("4. Save the .pt file as: weights/crocodile_detector.pt")
    print()
    print("Option 2: Train your own model")
    print("1. Collect crocodile images")
    print("2. Annotate using Roboflow or LabelImg")
    print("3. Train: yolo detect train data=crocodile.yaml epochs=50")
    print()
    print("Option 3: Use existing YOLOv8n (current placeholder)")
    print("The current placeholder is YOLOv8n which doesn't detect crocodiles.")
    print("=" * 60)
    
    # For now, keep the placeholder
    print("\n✅ Keeping placeholder model (YOLOv8n) at:", output_path)
    print("   Note: This model does NOT have crocodile in its classes.")
    print("   Replace it with a trained crocodile model for actual detection.")
    
    return False

if __name__ == "__main__":
    download_crocodile_model()
