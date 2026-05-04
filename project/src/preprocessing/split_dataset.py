import os
import shutil
import random
from tqdm import tqdm
import sys
from pathlib import Path

# Add project/src to sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PROJECT_ROOT, SPLIT_DIR

def split_yolo_dataset(source_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Consolidates existing YOLO splits and re-splits them into a clean 3-way distribution.
    """
    print(f"🔍 Analyzing source: {source_dir}")
    
    # 1. Collect all images from all subfolders in source
    all_images = []
    source_img_root = Path(source_dir) / "images"
    source_label_root = Path(source_dir) / "labels"
    
    for sub in ['train', 'val', 'test']:
        img_sub = source_img_root / sub
        if img_sub.exists():
            all_images.extend([img_sub / f for f in os.listdir(img_sub) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not all_images:
        print("❌ No images found in source!")
        return

    random.seed(42)
    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    print(f"📊 Found {total} total images.")
    print(f"📈 Re-splitting: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    for split_name, split_files in splits.items():
        img_dst = Path(SPLIT_DIR) / split_name / "images"
        label_dst = Path(SPLIT_DIR) / split_name / "labels"

        img_dst.mkdir(parents=True, exist_ok=True)
        label_dst.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(split_files, desc=f"📦 Processing {split_name}"):
            # Copy Image
            shutil.copy2(img_path, img_dst / img_path.name)
            
            # Find and copy corresponding label
            # Try to find label in any of the source label folders
            base_name = img_path.stem
            label_found = False
            for sub in ['train', 'val', 'test']:
                label_src = source_label_root / sub / f"{base_name}.txt"
                if label_src.exists():
                    shutil.copy2(label_src, label_dst / f"{base_name}.txt")
                    label_found = True
                    break
            
            if not label_found:
                # Optionally warn if label is missing
                pass

    print(f"\n✅ Data re-split successfully into: {SPLIT_DIR}")

if __name__ == "__main__":
    SOURCE = Path("/Users/HetviSheth/Flood_Prediction/project/dataset_yolo")
    split_yolo_dataset(SOURCE)
