import os
import cv2
import shutil
import random
from pathlib import Path
from tqdm import tqdm

class DatasetPreparer:
    """
    Validates, Resizes, and Splits raw imagery into YOLO format.
    """
    def __init__(self, raw_dir, output_dir, split_ratio=0.8, img_size=640):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.split_ratio = split_ratio
        self.img_size = img_size
        
        # Structure
        self.dirs = {
            'train_img': self.output_dir / "images" / "train",
            'val_img': self.output_dir / "images" / "val",
            'train_lbl': self.output_dir / "labels" / "train",
            'val_lbl': self.output_dir / "labels" / "val"
        }
        for d in self.dirs.values(): d.mkdir(parents=True, exist_ok=True)

    def is_corrupted(self, img_path):
        try:
            img = cv2.imread(str(img_path))
            if img is None: return True
            return False
        except:
            return True

    def process(self):
        images = [f for f in self.raw_dir.glob("*.jpg")] + [f for f in self.raw_dir.glob("*.png")]
        print(f"📦 Processing {len(images)} raw images...")
        
        valid_images = []
        for img_path in tqdm(images, desc="Validating"):
            if not self.is_corrupted(img_path):
                valid_images.append(img_path)
        
        random.shuffle(valid_images)
        split_idx = int(len(valid_images) * self.split_ratio)
        
        train_set = valid_images[:split_idx]
        val_set = valid_images[split_idx:]
        
        def move_and_resize(dataset, mode):
            img_dest = self.dirs[f'{mode}_img']
            lbl_dest = self.dirs[f'{mode}_lbl']
            
            for img_path in tqdm(dataset, desc=f"Moving {mode}"):
                # 1. Resize and Save Image
                img = cv2.imread(str(img_path))
                resized = cv2.resize(img, (self.img_size, self.img_size))
                cv2.imwrite(str(img_dest / img_path.name), resized)
                
                # 2. Handle Label (Look for matching .txt)
                lbl_path = img_path.with_suffix('.txt')
                if lbl_path.exists():
                    shutil.copy(lbl_path, lbl_dest / lbl_path.name)
                else:
                    # Create empty label file if missing (for background images)
                    open(lbl_dest / f"{img_path.stem}.txt", 'a').close()

        move_and_resize(train_set, 'train')
        move_and_resize(val_set, 'val')
        
        print(f"✨ Dataset Preparation Complete!")
        print(f"📂 Train: {len(train_set)} | Val: {len(val_set)}")

if __name__ == "__main__":
    preparer = DatasetPreparer(
        raw_dir="/Users/HetviSheth/Flood_Prediction/project/raw_dataset/images",
        output_dir="/Users/HetviSheth/Flood_Prediction/project/dataset_yolo"
    )
    preparer.process()
