import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms, models
from pathlib import Path

# -----------------------------
# PATH FIX
# -----------------------------
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import NUM_CLASSES, WEIGHTS_DIR, CLASSES, IMG_SIZE

# -----------------------------
# DEVICE
# -----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -----------------------------
# COLORS (BGR for OpenCV)
# -----------------------------
# 0: water, 1: road, 2: building, 3: vegetation
COLORS = [
    (255, 0, 0),      # Index 0: water (Blue)
    (0, 255, 0),      # Index 1: road (Green)
    (0, 0, 255),      # Index 2: building (Red)
    (0, 128, 255),    # Index 3: vegetation (Orange/Yellowish)
]

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model(device):
    # Same architecture as train.py
    model = models.segmentation.lraspp_mobilenet_v3_large(weights=None)
    model.classifier.low_classifier = nn.Conv2d(40, NUM_CLASSES, 1)
    model.classifier.high_classifier = nn.Conv2d(128, NUM_CLASSES, 1)

    weights_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
    if not os.path.exists(weights_path):
        print(f"❌ Weights not found at {weights_path}")
        return None

    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    model.to(device)
    model.eval()
    return model

# -----------------------------
# TRANSFORM
# -----------------------------
inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE[1], IMG_SIZE[0])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def color_mask(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(COLORS):
        colored[mask == i] = color
    return colored

def run_inference(img_path, model, device, output_dir):
    # Read
    img = cv2.imread(img_path)
    if img is None: return
    
    orig_h, orig_w = img.shape[:2]
    
    # Preprocess
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = inference_transform(rgb).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        out = model(tensor)["out"]
        pred = torch.argmax(out, dim=1)[0]
    
    # Process prediction
    pred_np = pred.cpu().numpy().astype(np.uint8)
    mask_colored = color_mask(pred_np)
    
    # Resize mask back to original size
    mask_colored = cv2.resize(mask_colored, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Overlay
    overlay = cv2.addWeighted(img, 0.6, mask_colored, 0.4, 0)
    
    # Save
    name = os.path.basename(img_path)
    save_path = os.path.join(output_dir, f"inf_{name}")
    cv2.imwrite(save_path, overlay)
    print(f"✅ Predicted: {name}")

def main():
    device = get_device()
    print(f"🚀 Using device: {device}")
    
    model = load_model(device)
    if model is None: return
    
    test_dir = os.path.join(project_root, "project", "test_samples")
    out_dir = os.path.join(project_root, "project", "outputs", "test_results")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        print(f"❌ Test directory {test_dir} is empty or missing.")
        return
        
    print(f"🔍 Testing on images in: {test_dir}")
    images = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    for img_name in images:
        path = os.path.join(test_dir, img_name)
        run_inference(path, model, device, out_dir)
        
    print(f"\n✨ DONE! Results saved to: {out_dir}")

if __name__ == "__main__":
    main()
