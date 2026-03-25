import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import pygame
from torchvision import transforms, models
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# PATH FIX
# -----------------------------
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DEVICE, NUM_CLASSES, MODEL_PATH, YOLO_MODEL_PATH, COLORS, INF_SIZE, ALERT_SOUND_PATH
from project.src.utils.logger import get_logger
from project.src.utils.ip_location import get_location
from project.src.utils.weather_api import get_rainfall

logger = get_logger("FloodCameraPro")

# Initialize Audio
pygame.mixer.init()

def play_alert():
    if os.path.exists(ALERT_SOUND_PATH):
        try:
            pygame.mixer.Sound(ALERT_SOUND_PATH).play()
        except: pass

def get_flood_analytics(mask):
    water_p = round((np.sum(mask == 0) / (mask.size - np.sum(mask == 255))) * 100, 2)
    severity = "HIGH" if water_p > 30 else "MEDIUM" if water_p > 10 else "LOW"
    return water_p, severity

def start_camera():
    yolo = YOLO(YOLO_MODEL_PATH)
    seg = models.segmentation.lraspp_mobilenet_v3_large(weights=None)
    seg.classifier.low_classifier = nn.Conv2d(40, NUM_CLASSES, 1)
    seg.classifier.high_classifier = nn.Conv2d(128, NUM_CLASSES, 1)
    if os.path.exists(MODEL_PATH):
        seg.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    seg.to(DEVICE).eval()
    
    # Context (Update once or periodically)
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)

    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        orig_h, orig_w = frame.shape[:2]
        
        # SEG
        with torch.no_grad():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize(INF_SIZE),
                transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(rgb).unsqueeze(0).to(DEVICE)
            pred = torch.argmax(seg(tensor)["out"], dim=1)[0].cpu().numpy().astype(np.uint8)
        
        water_p, severity = get_flood_analytics(pred)
        mask_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for i, color in enumerate(COLORS): mask_colored[pred == i] = color
        mask_colored = cv2.resize(mask_colored, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        display_frame = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)
        
        # YOLO
        results = yolo(frame, verbose=False)[0]
        for box in results.boxes:
            if float(box.conf[0]) > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # STATS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(display_frame, f"City: {city} | Rain: {rain}mm", (20, 30), 1, 1.5, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Water: {water_p}% | {severity}", (20, 65), 1, 1.5, (255, 255, 0), 2)
        if severity == "HIGH":
            cv2.putText(display_frame, "FLOOD ALERT", (orig_w//2-100, 50), 1, 2, (0, 0, 255), 3)
            if water_p > 30: play_alert()

        cv2.imshow("RAINWISE Pro Camera", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": start_camera()
