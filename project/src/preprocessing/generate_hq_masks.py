import os
import torch
import random
from ultralytics import FastSAM
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import json

def compute_iou_mask_box(mask, box):
    x1, y1, x2, y2 = map(int, box)
    box_mask = np.zeros_like(mask, dtype=np.uint8)
    box_mask[max(0, y1):min(mask.shape[0], y2), max(0, x1):min(mask.shape[1], x2)] = 1
    intersection = np.logical_and(mask, box_mask).sum()
    union = np.logical_or(mask, box_mask).sum()
    return intersection / (union + 1e-6)

def process_split(split_path, output_subdir, model, device, limit=20000, spot_check_n=50):
    split_path = Path(split_path)
    img_dir = split_path / "images"
    lbl_dir = split_path / "labels"
    out_dir = Path(output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    viz_dir = out_dir / "visual_health_check"
    viz_dir.mkdir(exist_ok=True)
    
    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])[:limit]
    print(f"🚀 Scaling {split_path.name.upper()}...")
    
    stats = {"total": 0, "accepted": 0, "rejected_iou": 0, "rejected_area": 0, "no_labels": 0}
    spot_check_indices = random.sample(range(len(images)), min(spot_check_n, len(images)))
    
    MIN_AREA = 200

    for idx, img_name in enumerate(tqdm(images)):
        stats["total"] += 1
        img_path = img_dir / img_name
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        
        if (out_dir / f"{img_path.stem}.png").exists():
            stats["accepted"] += 1
            continue

        if not lbl_path.exists():
            stats["no_labels"] += 1
            continue
            
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        img_area = h * w
        
        yolo_boxes = []
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                cx, cy, nw, nh = map(float, parts[1:])
                x1, y1 = int((cx - nw/2) * w), int((cy - nh/2) * h)
                x2, y2 = int((cx + nw/2) * w), int((cy + nh/2) * h)
                yolo_boxes.append({"coords": [x1, y1, x2, y2], "area": (x2-x1)*(y2-y1)})
        
        if not yolo_boxes:
            stats["no_labels"] += 1
            continue
            
        results = model(img_path, device=device, retina_masks=True, imgsz=640, conf=0.25, iou=0.9, verbose=False)[0]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        has_valid_mask = False
        if results.masks is not None:
            for mask in results.masks.data:
                mask_np = cv2.resize(mask.cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                if mask_np.sum() < MIN_AREA:
                    stats["rejected_area"] += 1
                    continue
                
                keep = False
                for bx in yolo_boxes:
                    adaptive_thresh = max(0.2, 0.4 * (bx["area"] / img_area))
                    if compute_iou_mask_box(mask_np, bx["coords"]) > adaptive_thresh:
                        keep = True
                        break
                
                if keep:
                    final_mask[mask_np > 0] = 1
                    has_valid_mask = True
                else:
                    stats["rejected_iou"] += 1
        
        if has_valid_mask:
            stats["accepted"] += 1
            cv2.imwrite(str(out_dir / f"{img_path.stem}.png"), final_mask)
            if idx in spot_check_indices:
                viz_img = img.copy()
                viz_img[final_mask == 1] = [0, 255, 0]
                cv2.imwrite(str(viz_dir / f"health_{img_name}"), cv2.addWeighted(img, 0.7, viz_img, 0.3, 0))

    with open(out_dir / "generation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

def generate_hq_masks():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = FastSAM('project/weights/FastSAM-s.pt')
    
    # PRIORITY: Process VAL first so we can start training ASAP with stable metrics
    process_split("project/dataset_split/val", "project/dataset_split/val/masks_hq", model, device)
    process_split("project/dataset_split/train", "project/dataset_split/train/masks_hq", model, device)

if __name__ == "__main__":
    generate_hq_masks()
