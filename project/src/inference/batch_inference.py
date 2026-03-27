import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import csv
import json
from tqdm import tqdm
from torchvision import transforms, models
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# PATH FIX
# -----------------------------
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DEVICE, NUM_CLASSES, MODEL_PATH, YOLO_MODEL_PATH, COLORS, INF_SIZE, OUTPUT_DIR
from project.src.utils.logger import get_logger
from project.src.utils.ip_location import get_location
from project.src.utils.weather_api import get_rainfall
from project.src.utils.area import calculate_water_area
from project.src.utils.risk import get_flood_risk

logger = get_logger("BatchFloodPro")

def load_models():
    # Load YOLO once as per requirements
    yolo = YOLO("yolov8n.pt")
    seg = models.segmentation.lraspp_mobilenet_v3_large(weights=None)
    seg.classifier.low_classifier = nn.Conv2d(40, NUM_CLASSES, 1)
    seg.classifier.high_classifier = nn.Conv2d(128, NUM_CLASSES, 1)
    if os.path.exists(MODEL_PATH):
        seg.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    seg.to(DEVICE).eval()
    return yolo, seg

def process_folder(input_dir):
    if not os.path.exists(input_dir): return
    out_dir = os.path.join(OUTPUT_DIR, "batch_flood_results")
    os.makedirs(out_dir, exist_ok=True)
    
    yolo, seg = load_models()
    images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))]
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)

    report_data = []

    for img_name in tqdm(images, desc="Pro Batch Analysis"):
        path = os.path.join(input_dir, img_name)
        img = cv2.imread(path)
        if img is None: continue
        orig_h, orig_w = img.shape[:2]
        
        # 1. YOLO (PRIMARY)
        yolo_results = yolo(img, verbose=False)[0]
        obj_count = len(yolo_results.boxes)
        
        # 2. SEGMENTATION (SECONDARY)
        with torch.no_grad():
            tensor = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize(INF_SIZE), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
            pred = torch.argmax(seg(tensor)["out"], dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # 3. ANALYTICS
        water_p = calculate_water_area(pred)
        
        # 4. YOLO-PRIMARY HYBRID LOGIC
        risk = get_flood_risk(water_p, obj_count)
        
        mask_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for i, color in enumerate(COLORS): mask_colored[pred == i] = color
        mask_colored = cv2.resize(mask_colored, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        combined = cv2.addWeighted(img, 0.6, mask_colored, 0.4, 0)

        # 5. OVERLAYS - Simplified & Professional
        cv2.putText(combined, f"Water: {water_p:.1f}% | Objects: {obj_count} | Risk: {risk}", (20, 40), 1, 2, (0, 255, 0), 2)
        
        if risk == "FLOOD":
            cv2.putText(combined, "FLOOD ALERT", (orig_w//2-150, orig_h-50), 1, 3, (0, 0, 255), 4)

        cv2.imwrite(os.path.join(out_dir, f"pro_{img_name}"), combined)
        report_data.append({"file": img_name, "water": water_p, "risk": risk, "city": city, "lat": lat, "lon": lon, "rain": rain})

    # Save Reports
    report_dir = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "pro_batch_report.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "water", "risk", "city", "lat", "lon", "rain"])
        writer.writeheader()
        writer.writerows(report_data)
        
    with open(os.path.join(report_dir, "pro_batch_report.json"), "w") as f:
        json.dump(report_data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) > 1: process_folder(sys.argv[1])
