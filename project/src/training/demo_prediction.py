import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.segformer_model import SegFormerFlood
from models.custom_dataset import FloodCustomDataset

def run_demo():
    # 1. SETUP
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATASET_ROOT = PROJECT_ROOT / "dataset_split"
    WEIGHTS_PATH = PROJECT_ROOT / "weights" / "rainwise_v3_1_demo.pth"
    
    if not WEIGHTS_PATH.exists():
        print(f"❌ ERROR: Model not found at {WEIGHTS_PATH}")
        return

    # 2. LOAD MODEL
    model = SegFormerFlood(num_classes=2, in_channels=6).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print("✅ Model Loaded Successfully!")

    # 3. GET SAMPLE IMAGE
    val_dataset = FloodCustomDataset(
        str(DATASET_ROOT / "val" / "images"), 
        masks_dir=str(DATASET_ROOT / "val" / "masks_hq"), 
        balance=False
    )
    
    # Pick a random sample from validation
    idx = np.random.randint(len(val_dataset))
    data, mask, _ = val_dataset[idx]
    
    # 4. INFERENCE
    with torch.no_grad():
        input_tensor = data.unsqueeze(0).to(device)
        output = model(input_tensor)
        
        # Process output
        pred_probs = torch.softmax(output[0], dim=0)[1]
        pred_mask = (pred_probs.cpu().numpy() > 0.5).astype(np.uint8) * 255

    # 5. VISUALIZE
    img_display = (data[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
    
    gt_mask = (mask.numpy() * 255).astype(np.uint8)
    
    # Create side-by-side
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Real Image")
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth (Water)")
    plt.imshow(gt_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("AI Prediction (SegFormer)")
    plt.imshow(pred_mask, cmap='jet') # Jet highlights the detection
    plt.axis('off')
    
    save_path = PROJECT_ROOT / "demo_result.png"
    plt.savefig(save_path)
    print(f"🎉 Demo image saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_demo()
