import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.segformer_model import SegFormerFlood

def run_video_demo(video_path=None):
    # 1. SETUP
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    WEIGHTS_PATH = PROJECT_ROOT / "weights" / "rainwise_v3_1_demo.pth"
    
    if not WEIGHTS_PATH.exists():
        print(f"❌ ERROR: Model weights not found!")
        return

    # 2. LOAD MODEL
    model = SegFormerFlood(num_classes=2, in_channels=6).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print("✅ Full Intelligence Engine Ready!")

    # 3. OPEN VIDEO
    if video_path is None:
        test_vids = list(PROJECT_ROOT.glob("test_videos/*.mp4"))
        if not test_vids:
            print("❌ No test videos found!")
            return
        video_path = str(test_vids[0])

    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 4. MULTI-SPECTRAL PRE-PROCESSING
        img_small = cv2.resize(frame, (256, 256))
        rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Calculate NDWI (Green-Red index)
        g, r = rgb[:,:,1], rgb[:,:,0]
        ndwi = ((g - r) / (g + r + 1e-6) + 1) / 2
        
        # Calculate Texture (Sobel)
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture = cv2.normalize(np.sqrt(sobel_x**2 + sobel_y**2), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        
        # Calculate Elevation Gradient
        h, w = 256, 256
        elevation = np.linspace(1, 0, h).reshape(h, 1).repeat(w, axis=1).astype(np.float32)
        
        # Assemble 6-Channel Input
        six_channel = np.zeros((h, w, 6), dtype=np.float32)
        six_channel[:, :, :3] = rgb
        six_channel[:, :, 3] = ndwi
        six_channel[:, :, 4] = texture
        six_channel[:, :, 5] = elevation
        
        input_tensor = torch.from_numpy(six_channel).permute(2, 0, 1).unsqueeze(0).to(device)

        # 5. INFERENCE
        with torch.no_grad():
            output = model(input_tensor)
            pred_probs = torch.softmax(output[0], dim=0)[1]
            # Using 0.7 for balance
            mask = (pred_probs.cpu().numpy() > 0.7).astype(np.uint8) * 255

        # 6. OVERLAY
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        overlay = frame.copy()
        overlay[mask_resized > 0] = [0, 0, 255] # Red highlight
        
        combined = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.putText(combined, "RAINWISE V3.1: MULTI-SPECTRAL AI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('RAINWISE Flood Intelligence', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video_demo()
