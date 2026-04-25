import os
import cv2
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np

class DatasetCleaner:
    """
    Cleans a raw dataset by removing blurry frames and near-duplicates.
    """
    def __init__(self, images_dir):
        self.images_dir = Path(images_dir)

    def is_blurry(self, image, threshold=100.0):
        """Variance of Laplacian for blur detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    def clean_dataset(self, blur_threshold=100.0, hash_threshold=5):
        """
        Removes blurry images and duplicates using perceptual hashing.
        """
        images = list(self.images_dir.glob("*.jpg"))
        print(f"🧹 Cleaning {len(images)} images...")
        
        hashes = {} # {hash: filename}
        removed_blur = 0
        removed_duplicate = 0
        
        for img_path in tqdm(images):
            try:
                # 1. Blur Detection
                img = cv2.imread(str(img_path))
                if img is None: continue
                
                if self.is_blurry(img, blur_threshold):
                    os.remove(img_path)
                    removed_blur += 1
                    continue
                
                # 2. Duplicate Detection (PHash)
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                h = imagehash.phash(pil_img)
                
                # Check for near matches
                duplicate = False
                for existing_hash in hashes:
                    if h - existing_hash <= hash_threshold:
                        os.remove(img_path)
                        removed_duplicate += 1
                        duplicate = True
                        break
                
                if not duplicate:
                    hashes[h] = img_path
                    
            except Exception as e:
                print(f"⚠️ Error processing {img_path.name}: {e}")

        print(f"✨ Cleanup Complete!")
        print(f"🗑️ Removed Blurry: {removed_blur}")
        print(f"🗑️ Removed Duplicates: {removed_duplicate}")
        print(f"✅ Remaining: {len(list(self.images_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    cleaner = DatasetCleaner(images_dir="/Users/HetviSheth/Flood_Prediction/project/raw_dataset/images")
    cleaner.clean_dataset()
