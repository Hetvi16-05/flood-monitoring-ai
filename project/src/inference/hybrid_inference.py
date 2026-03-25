import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import json
import csv
from datetime import datetime
from torchvision import transforms, models
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# PATH FIX
# -----------------------------
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import NUM_CLASSES, WEIGHTS_DIR, CLASSES, IMG_SIZE, COLORS
from project.src.utils.logger import get_logger
from project.src.utils.ip_location import get_location
from project.src.utils.weather_api import get_rainfall
from project.src.utils.email_alert import send_flood_alert

logger = get_logger("FloodInferencePro")

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# -----------------------------
# ANALYTICS
# -----------------------------
def get_flood_analytics(mask):
    water_pixels = np.sum(mask == 0)
    total_pixels = mask.size - np.sum(mask == 255)
    if total_pixels == 0: return 0.0, "LOW"
    water_p = (water_pixels / total_pixels) * 100
    severity = "HIGH" if water_p > 30 else "MEDIUM" if water_p > 10 else "LOW"
    return round(water_p, 2), severity

def save_report(filename, water_p, severity, objs, latency, rain, lat, lon, city):
    report_dir = os.path.join(project_root, "project", "outputs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    csv_path = os.path.join(report_dir, "report.csv")
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["Timestamp", "Filename", "Water%", "Severity", "Objects", "Latency_ms", "Rain_mm", "Lat", "Lon", "City"])
        writer.writerow([timestamp, filename, water_p, severity, objs, latency, rain, lat, lon, city])
    
    json_path = os.path.join(report_dir, f"report_{os.path.basename(filename)}.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp, "file": filename, "water_percent": water_p,
            "severity": severity, "objects": objs, "latency": latency,
            "rainfall": rain, "lat": lat, "lon": lon, "city": city
        }, f, indent=4)

# -----------------------------
# LOAD MODELS
# -----------------------------
def load_models(device):
    yolo = YOLO("yolov8n.pt") 
    model = models.segmentation.lraspp_mobilenet_v3_large(weights=None)
    model.classifier.low_classifier = nn.Conv2d(40, NUM_CLASSES, 1)
    model.classifier.high_classifier = nn.Conv2d(128, NUM_CLASSES, 1)
    weights = os.path.join(WEIGHTS_DIR, "best_model.pth")
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()
    return yolo, model

def run_hybrid_inference(img_path, yolo, seg, device, out_dir):
    img = cv2.imread(img_path)
    if img is None: return
    orig_h, orig_w = img.shape[:2]
    
    # Contextual Intel
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)
    
    # 1. SEG
    t0 = time.time()
    with torch.no_grad():
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((256, 256)),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(rgb).unsqueeze(0).to(device)
        pred = torch.argmax(seg(tensor)["out"], dim=1)[0].cpu().numpy().astype(np.uint8)
    
    water_p, severity = get_flood_analytics(pred)
    t_seg = (time.time() - t0) * 1000

    mask_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(COLORS): mask_colored[pred == i] = color
    mask_colored = cv2.resize(mask_colored, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    combined = cv2.addWeighted(img, 0.6, mask_colored, 0.4, 0)
    
    # 2. YOLO
    t1 = time.time()
    results = yolo(img, verbose=False)[0]
    num_objs = len(results.boxes)
    for box in results.boxes:
        if float(box.conf[0]) > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(combined, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # 3. OVERLAY
    stats = [
        f"City: {city}", f"GPS: {lat:.4f}, {lon:.4f}", f"Rain: {rain}mm",
        f"Water: {water_p}%", f"Severity: {severity}", f"Latency: {t_seg:.1f}ms"
    ]
    for i, txt in enumerate(stats):
        cv2.putText(combined, txt, (20, 40 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if severity == "HIGH":
        cv2.putText(combined, "FLOOD ALERT", (orig_w//2-100, orig_h-50), 1, 3, (0,0,255), 3)
        send_flood_alert(f"Level: {water_p}% | Objects: {num_objs}", city)

    name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(out_dir, f"pro_{name}"), combined)
    save_report(name, water_p, severity, num_objs, t_seg, rain, lat, lon, city)

def main():
    device = get_device()
    yolo, seg = load_models(device)
    test_dir = os.path.join(project_root, "project", "test_samples")
    out_dir = os.path.join(project_root, "project", "outputs", "flood_results")
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    if os.path.exists(test_dir):
        for img in [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png"))]:
            run_hybrid_inference(os.path.join(test_dir, img), yolo, seg, device, out_dir)

if __name__ == "__main__": main()
