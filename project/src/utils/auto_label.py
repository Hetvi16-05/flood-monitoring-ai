import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from ultralytics import FastSAM

from config import CLEAN_DIR, IMG_SIZE, CLASSES, PROJECT_ROOT

# ---------------- DEVICE ----------------

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = get_device()
print(f"🚀 Using Device: {device}")

# ---------------- LABEL ----------------

def get_label(name):
    for i, c in enumerate(CLASSES):
        if name.startswith(c):
            return i
    return 0

# ---------------- GLOBAL MODEL ----------------

model = None

def init_worker():
    global model
    weights_path = os.path.join(PROJECT_ROOT, "project", "weights", "FastSAM-s.pt")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    # Loads model (downloads if missing)
    model = FastSAM("FastSAM-s.pt" if not os.path.exists(weights_path) else weights_path)
    model.to(device)

# ---------------- PROCESS ----------------

def process_image(name):
    img_dir = os.path.join(CLEAN_DIR, "images")
    label_dir = os.path.join(CLEAN_DIR, "labels")
    mask_dir = os.path.join(CLEAN_DIR, "masks")

    img_path = os.path.join(img_dir, name)
    base_name = name.rsplit('.', 1)[0]
    
    label_path = os.path.join(label_dir, base_name + ".txt")
    mask_path = os.path.join(mask_dir, base_name + ".png")

    # If both exist, skip
    if os.path.exists(label_path) and os.path.exists(mask_path):
        return

    img = cv2.imread(img_path)
    if img is None:
        return

    img = cv2.resize(img, IMG_SIZE)
    base_label = get_label(name)

    # FastSAM Predict
    results = model.predict(
        source=img,
        device=device,
        conf=0.4,
        iou=0.9,
        imgsz=IMG_SIZE[0],
        retina_masks=True,
        verbose=False
    )
    
    if not results or not results[0].masks:
        return

    # 1. Save YOLO Polygons (.txt)
    polygons = results[0].masks.xyn
    all_yolo_lines = []
    for poly in polygons:
        if len(poly) < 3: continue
        line = f"{base_label} " + " ".join([f"{c:.6f}" for coord in poly for c in coord])
        all_yolo_lines.append(line)

    if all_yolo_lines:
        os.makedirs(label_dir, exist_ok=True)
        with open(label_path, "w") as f:
            f.write("\n".join(all_yolo_lines))

    # 2. Save Classification Mask (.png) for DeepLabV3
    # We ensure every pixel has a valid class ID (0, 1, 2, 3)
    # Background defaults to the base_label of the image folder
    masks_data = results[0].masks.data # [N, H, W]
    
    # Initialize with base_label instead of 255 (ignore)
    final_mask = np.full((IMG_SIZE[1], IMG_SIZE[0]), base_label, dtype=np.uint8)
    
    if masks_data is not None and len(masks_data) > 0:
        # Use torch.any to get combined binary mask of detected objects
        combined_binary = torch.any(masks_data, dim=0).cpu().numpy().astype(np.uint8)
        # We ensure it matches expected size
        if combined_binary.shape != (IMG_SIZE[1], IMG_SIZE[0]):
            combined_binary = cv2.resize(combined_binary, (IMG_SIZE[0], IMG_SIZE[1]), interpolation=cv2.INTER_NEAREST)
        
        # Detected areas are set to base_label (redundant here but keeps logic consistent)
        final_mask[combined_binary == 1] = base_label
        
    os.makedirs(mask_dir, exist_ok=True)
    cv2.imwrite(mask_path, final_mask)

# ---------------- AUTO LABEL ----------------

def auto_label():
    img_dir = os.path.join(CLEAN_DIR, "images")
    os.makedirs(os.path.join(CLEAN_DIR, "labels"), exist_ok=True)
    os.makedirs(os.path.join(CLEAN_DIR, "masks"), exist_ok=True)

    images = [
        f for f in os.listdir(img_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"Total images to process: {len(images)}")

    # Faster without too many processes on MPS
    workers = 1 if device == "mps" else max(2, cpu_count() // 2)
    print(f"Using {workers} workers")

    with Pool(
        workers,
        initializer=init_worker
    ) as p:
        list(
            tqdm(
                p.imap(process_image, images),
                total=len(images)
            )
        )

if __name__ == "__main__":
    auto_label()