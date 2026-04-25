import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class FloodCustomDataset(Dataset):
    """
    Efficient Dataset for 6-channel FloodNet training.
    Generates NDWI, Texture, and Elevation channels at runtime.
    """
    def __init__(self, images_dir, masks_dir, transform=None, img_size=(256, 256)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Support both .jpg and .png images
        self.samples = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.png'))]
        print(f"📊 Dataset initialized with {len(self.samples)} samples from {self.images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        img_name = self.samples[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # 1. Load and Resize Image
        img_path = str(self.images_dir / img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            # Return zero tensors if image fails to load
            return torch.zeros((6, self.img_size[1], self.img_size[0])), torch.zeros(self.img_size, dtype=torch.long), []
            
        frame = cv2.resize(frame, self.img_size)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 2. Compute NDWI ((G-R)/(G+R))
        g, r = rgb[:,:,1], rgb[:,:,0]
        ndwi = (g - r) / (g + r + 1e-6)
        ndwi = (ndwi + 1) / 2 # Scale to 0-1
        
        # 3. Compute Texture (Sobel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture = np.sqrt(sobelx**2 + sobely**2)
        texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)
        
        # 4. Generate Mock Elevation (Top-Down Gradient)
        h, w = self.img_size
        elevation = np.linspace(1, 0, h).reshape(h, 1).repeat(w, axis=1).astype(np.float32)
        
        # 5. Assemble 6-Channel Tensor
        six_channel = np.zeros((h, w, 6), dtype=np.float32)
        six_channel[:, :, :3] = rgb
        six_channel[:, :, 3] = ndwi
        six_channel[:, :, 4] = texture
        six_channel[:, :, 5] = elevation
        
        # 6. Load Mask
        mask_path = self.masks_dir / f"{base_name}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros(self.img_size, dtype=np.uint8)
            
        # Convert to Tensors
        data_tensor = torch.from_numpy(six_channel).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()
        
        return data_tensor, mask_tensor, [] # No detection labels needed for primary segmentation
