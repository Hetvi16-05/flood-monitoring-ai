import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import csv
import pygame
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

from config import DEVICE, NUM_CLASSES, MODEL_PATH, YOLO_MODEL_PATH, COLORS, INF_SIZE, OUTPUT_DIR, ALERT_SOUND_PATH
from project.src.utils.logger import get_logger
from project.src.utils.ip_location import get_location
from project.src.utils.weather_api import get_rainfall

logger = get_logger("FloodVideoPro")
pygame.mixer.init()

def play_alert():
    if os.path.exists(ALERT_SOUND_PATH):
        try: pygame.mixer.Sound(ALERT_SOUND_PATH).play()
        except: pass

def get_flood_analytics(mask):
    water_p = round((np.sum(mask == 0) / (mask.size - np.sum(mask == 255))) * 100, 1)
    severity = "HIGH" if water_p > 30 else "MEDIUM" if water_p > 10 else "LOW"
    return water_p, severity

def process_video(input_path):
    yolo = YOLO(YOLO_MODEL_PATH)
    seg = models.segmentation.lraspp_mobilenet_v3_large(weights=None).to(DEVICE)
    if os.path.exists(MODEL_PATH): seg.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    seg.eval()
    
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)

    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps, total = cap.get(5), int(cap.get(7))

    out_dir = os.path.join(OUTPUT_DIR, "video_results")
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    save_path = os.path.join(out_dir, f"pro_{os.path.basename(input_path)}")
    out_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    results_data = []
    pbar = tqdm(total=total, desc="Pro Video Analysis")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        with torch.no_grad():
            tensor = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize(INF_SIZE), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
            pred = torch.argmax(seg(tensor)["out"], dim=1)[0].cpu().numpy().astype(np.uint8)
        
        water_p, severity = get_flood_analytics(pred)
        mask_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for i, color in enumerate(COLORS): mask_colored[pred == i] = color
        mask_colored = cv2.resize(mask_colored, (width, height), interpolation=cv2.INTER_NEAREST)
        display_frame = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)
        
        # Overlays
        cv2.putText(display_frame, f"{city} | {lat:.2f}, {lon:.2f} | Rain: {rain}mm", (20, 30), 1, 1.5, (0,255,0), 2)
        cv2.putText(display_frame, f"Water: {water_p}% | {severity}", (20, 65), 1, 1.5, (255,255,0), 2)
        if severity == "HIGH": 
            cv2.putText(display_frame, "FLOOD ALERT", (width//2-100, height-50), 1, 2, (0,0,255), 3)
            if water_p > 30: play_alert()

        out_video.write(display_frame)
        results_data.append([frame_idx, water_p, severity, city, lat, lon, rain])
        pbar.update(1)
        frame_idx += 1

    cap.release()
    out_video.release()
    pbar.close()
    
    # Save Pro Report
    csv_path = os.path.join(OUTPUT_DIR, "reports", f"pro_{os.path.basename(input_path)}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Water%", "Severity", "City", "Lat", "Lon", "Rain_mm"])
        writer.writerows(results_data)

if __name__ == "__main__":
    if len(sys.argv) > 1: process_video(sys.argv[1])
