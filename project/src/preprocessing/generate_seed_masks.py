import cv2
import numpy as np
import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "project" / "dataset_custom" / "images"
MASK_DIR = ROOT / "project" / "dataset_custom" / "masks_8c"

# Classes:
# 0: flood_water, 1: shallow_water, 2: deep_water, 3: debris, 
# 4: road, 5: building, 6: vegetation, 7: emergency_objects

def generate_seed_masks():
    os.makedirs(MASK_DIR, exist_ok=True)
    images = os.listdir(IMAGE_DIR)
    
    for img_name in images:
        img_path = IMAGE_DIR / img_name
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Urban Road Cars image logic
        if "urban_road_cars" in img_name:
            # Buildings on sides
            mask[:, :w//4] = 5
            mask[:, 3*w//4:] = 5
            # Water on right half of the street
            mask[h//2:, w//4:3*w//4] = 0
            # Road on left half
            mask[:h//2, w//4:3*w//4] = 4
            # Debris foreground
            mask[h-50:, w//3:2*w//3] = 3

        # 2. Rural Rescue Boat logic
        elif "rural_rescue_boat" in img_name:
            # Vegetation everywhere
            mask[:] = 6
            # Water covering most parts
            mask[h//3:, :] = 0
            # Boat in middle
            mask[h//2:h//2+40, w//2-40:w//2+40] = 7

        # 3. Bridge Submerged logic
        elif "bridge_submerged" in img_name:
            # Water river
            mask[:] = 2 # deep water
            # Bridge in center
            mask[h//2-30:h//2+30, :] = 4 # treating bridge as road/infrastructure

        save_path = MASK_DIR / f"{os.path.splitext(img_name)[0]}.png"
        cv2.imwrite(str(save_path), mask)
        print(f"🎭 Seed mask generated for {img_name}")

if __name__ == "__main__":
    generate_seed_masks()
