import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Paths
project_root = "/Users/HetviSheth/RAINWISE_CV_VADODARA"
mask_dir = os.path.join(project_root, "project", "dataset_clean", "masks")
img_dir = os.path.join(project_root, "project", "dataset_clean", "images")

# Classes (from config.py)
CLASSES = ["water", "road", "building", "vegetation"]

def get_label(name):
    for i, c in enumerate(CLASSES):
        if name.startswith(c):
            return i
    return 0

def migrate():
    print("🔄 Migrating masks to 'Ignore Background' format (255)...")
    
    masks = [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")]
    print(f"Total masks to process: {len(masks)}")

    for m_name in tqdm(masks):
        mask_path = os.path.join(mask_dir, m_name)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        # Determine correct label
        label = get_label(m_name)
        
        # Create new mask with 255 everywhere
        new_mask = np.full(mask.shape, 255, dtype=np.uint8)
        
        # Set pixels that were either the specific label or legacy 255
        object_mask = (mask == label) | (mask == 255)
        new_mask[object_mask] = label
        
        # Save back
        cv2.imwrite(mask_path, new_mask)

    print("✨ Migration complete! All non-object areas are now set to 255 (ignore).")

if __name__ == "__main__":
    migrate()
