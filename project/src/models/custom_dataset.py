import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import cv2

class FloodCustomDataset(Dataset):
    """
    Advanced Dataset for 6-channel Swin Transformer training.
    Supports:
    1. Runtime 6-channel feature generation (NDWI, Sobel, Elevation).
    2. Auto-conversion: Converts YOLO .txt labels to masks if .png masks are missing.
    """
    def __init__(self, images_dir, masks_dir=None, labels_dir=None, img_size=(256, 256)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        # Try to infer labels_dir if not provided (parallel to images)
        self.labels_dir = Path(labels_dir) if labels_dir else (self.images_dir.parent / "labels")
        self.img_size = img_size
        
        self.samples = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.png'))]
        print(f"📊 Dataset initialized with {len(self.samples)} samples.")
        if not self.masks_dir or not os.path.exists(self.masks_dir):
            print(f"💡 No mask directory found. Using YOLO labels from {self.labels_dir} as proxy.")

    def __len__(self):
        return len(self.samples)

    def yolo_to_mask(self, label_path, h, w):
        """Convert YOLO bounding boxes to a segmentation mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        if not label_path.exists():
            return mask
            
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                cls_id = int(parts[0])
                # Convert normalized xywh to pixel coordinates
                cx, cy, nw, nh = map(float, parts[1:])
                x1 = int((cx - nw/2) * w)
                y1 = int((cy - nh/2) * h)
                x2 = int((cx + nw/2) * w)
                y2 = int((cy + nh/2) * h)
                
                # Draw filled rectangle as mask proxy
                cv2.rectangle(mask, (x1, y1), (x2, y2), cls_id, -1)
        return mask

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # 1. Load and Resize Image
        img_path = str(self.images_dir / img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            return torch.zeros((6, self.img_size[1], self.img_size[0])), torch.zeros(self.img_size, dtype=torch.long), []
            
        frame = cv2.resize(frame, self.img_size)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 2. Compute NDWI
        g, r = rgb[:,:,1], rgb[:,:,0]
        ndwi = (g - r) / (g + r + 1e-6)
        ndwi = (ndwi + 1) / 2
        
        # 3. Compute Texture (Sobel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture = np.sqrt(sobelx**2 + sobely**2)
        texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)
        
        # 4. Generate Mock Elevation
        h, w = self.img_size
        elevation = np.linspace(1, 0, h).reshape(h, 1).repeat(w, axis=1).astype(np.float32)
        
        # 5. Assemble 6-Channel Tensor
        six_channel = np.zeros((h, w, 6), dtype=np.float32)
        six_channel[:, :, :3] = rgb
        six_channel[:, :, 3] = ndwi
        six_channel[:, :, 4] = texture
        six_channel[:, :, 5] = elevation
        
        # 6. Load or Generate Mask
        mask = None
        # Try high-res .png mask first
        if self.masks_dir:
            mask_path = self.masks_dir / f"{base_name}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Fallback: Generate from YOLO labels if mask is still None
        if mask is None:
            label_path = self.labels_dir / f"{base_name}.txt"
            mask = self.yolo_to_mask(label_path, h, w)
            
        # Convert to Tensors
        data_tensor = torch.from_numpy(six_channel).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()
        
        return data_tensor, mask_tensor, []
