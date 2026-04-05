import os
import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path to allow absolute imports
# project/src/preprocessing/preprocess.py -> project_root is 3 levels up
project_root = str(Path(__file__).resolve().parents[3])
print(f"DEBUG: __file__ = {__file__}")
print(f"DEBUG: project_root = {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"DEBUG: Inserted {project_root} into sys.path")

from config import RAW_DIR, CLEAN_DIR, IMG_SIZE
from preprocessing.image_processing import resize_image, normalize_image
from preprocessing.noise_reduction import bilateral_filter
from preprocessing.blur import gaussian_blur
from preprocessing.edge_detection import canny_edge

def process_pipeline(image_path, output_path, apply_edge=False):
    """Execution of the full preprocessing pipeline on a single image."""
    try:
        # 1. Read
        img = cv2.imread(image_path)
        if img is None:
            return False

        # 2. Resize
        img = resize_image(img, IMG_SIZE)

        # 3. Denoise (Preserving Edges)
        img = bilateral_filter(img)

        # 4. Blur - REMOVED for High Accuracy (use training augmentation instead)
        # 5. Optional Edge Detection - REMOVED (model learns edges better from RGB)

        # 6. Normalize (Note: For saving as image, we keep 0-255. For training, we normalize in Dataset class)
        # img = normalize_image(img) 

        # Save to destination
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    print("🧹 Starting Preprocessing Pipeline...")
    
    images_out_dir = os.path.join(CLEAN_DIR, "images")
    os.makedirs(images_out_dir, exist_ok=True)

    # Process each class in raw_dataset
    for cls in os.listdir(RAW_DIR):
        cls_path = os.path.join(RAW_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        print(f"\nProcessing class: {cls}")
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for f in tqdm(files, desc=f"  Cleaning {cls}"):
            src_path = os.path.join(cls_path, f)
            # Prepend class name to filename to avoid collisions and keep track of labels
            dst_name = f"{cls}_{f}"
            dst_path = os.path.join(images_out_dir, dst_name)
            
            process_pipeline(src_path, dst_path)

    print("\n✅ Preprocessing Complete! Clean images saved to:", images_out_dir)

if __name__ == "__main__":
    main()
