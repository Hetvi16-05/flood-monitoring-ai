import os
import shutil
import random
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import CLEAN_DIR, SPLIT_DIR

def split_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Physically split images and labels from dataset_clean to dataset_split."""
    
    img_src = os.path.join(CLEAN_DIR, "images")
    label_src = os.path.join(CLEAN_DIR, "labels")
    mask_src = os.path.join(CLEAN_DIR, "masks")
    
    # Ensure source images exist
    if not os.path.exists(img_src):
        print(f"❌ Source image directory not found: {img_src}")
        return

    # Get all images
    images = [f for f in os.listdir(img_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42)  # For reproducibility
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    print(f"📊 Total images: {total}")
    print(f"📈 Split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    for split_name, split_files in splits.items():
        img_dst = os.path.join(SPLIT_DIR, split_name, "images")
        label_dst = os.path.join(SPLIT_DIR, split_name, "labels")
        # masks are usually for DeepLabV3, YOLO uses labels. We split both.
        mask_dst = os.path.join(SPLIT_DIR, split_name, "masks")

        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(label_dst, exist_ok=True)
        os.makedirs(mask_dst, exist_ok=True)

        for f in tqdm(split_files, desc=f"  Moving to {split_name}"):
            # Copy Image
            shutil.copy2(os.path.join(img_src, f), os.path.join(img_dst, f))
            
            # Copy Label (if exists)
            base_name = f.rsplit('.', 1)[0]
            label_f = base_name + ".txt"
            if os.path.exists(os.path.join(label_src, label_f)):
                shutil.copy2(os.path.join(label_src, label_f), os.path.join(label_dst, label_f))
                
            # Copy Mask (if exists)
            mask_f = base_name + ".png"
            if os.path.exists(os.path.join(mask_src, mask_f)):
                shutil.copy2(os.path.join(mask_src, mask_f), os.path.join(mask_dst, mask_f))

    print("\n✅ Dataset Split Complete!")
    print(f"📂 Results saved in: {SPLIT_DIR}")

if __name__ == "__main__":
    # Check if a split already exists and warn (optional)
    split_dataset()
