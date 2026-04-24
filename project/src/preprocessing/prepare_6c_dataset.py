import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Path Configuration
ROOT = Path(__file__).resolve().parents[3]
RAW_IMAGES = ROOT / "project" / "dataset_custom" / "images"
PROCESSED_6C = ROOT / "project" / "dataset_custom" / "processed_6c"
ELEV_DIR = ROOT / "project" / "dataset_custom" / "elevation_dem"
IMG_SIZE = (512, 512)

def generate_6_channel_data():
    """
    Transforms RGB images into 6-channel tensors:
    [R, G, B, WaterIndex, Texture, Elevation]
    """
    os.makedirs(PROCESSED_6C, exist_ok=True)
    images = [f for f in os.listdir(RAW_IMAGES) if f.lower().endswith(('.jpg', '.png'))]
    
    if not images:
        print(f"⚠️ No images found in {RAW_IMAGES}. Please add your flood imagery first.")
        return

    print(f"⚙️  Processing {len(images)} images into 6-channel tensors...")
    
    for img_name in tqdm(images):
        # 1. Load RGB
        img_path = str(RAW_IMAGES / img_name)
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, IMG_SIZE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 2. Water Index (Modified NDWI: (G - R) / (G + R))
        g, r = rgb[:,:,1], rgb[:,:,0]
        ndwi = (g - r) / (g + r + 1e-6)
        ndwi = (ndwi + 1) / 2 # Scale to 0-1
        
        # 3. Texture (Sobel Magnitude)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture = np.sqrt(sobelx**2 + sobely**2)
        texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)
        
        # 4. Elevation (Load from DEM if exists, else Mock Gradient)
        elev_path = ELEV_DIR / f"{os.path.splitext(img_name)[0]}.npy"
        if elev_path.exists():
            elevation = np.load(elev_path)
            elevation = cv2.resize(elevation, IMG_SIZE)
        else:
            # Mock: Assume top is higher ground, bottom is lower (typical for flood logic)
            h, w = IMG_SIZE
            elevation = np.linspace(1, 0, h).reshape(h, 1).repeat(w, axis=1)
            
        # 5. Assemble 6-Channel Array
        six_channel = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 6), dtype=np.float32)
        six_channel[:, :, :3] = rgb
        six_channel[:, :, 3] = ndwi
        six_channel[:, :, 4] = texture
        six_channel[:, :, 5] = elevation
        
        # 6. Save as NPY for fast training access
        save_name = os.path.splitext(img_name)[0] + ".npy"
        np.save(str(PROCESSED_6C / save_name), six_channel)

if __name__ == "__main__":
    generate_6_channel_data()
