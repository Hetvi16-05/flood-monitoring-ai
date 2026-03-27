import os
import sys
import cv2
import time
from pathlib import Path

# -----------------------------
# PATH FIX (Standardized)
# -----------------------------
# File is in project/src/inference/hybrid_camera.py
project_root = Path(__file__).resolve().parents[3]
src_path = project_root / "project" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from config import ALERT_SOUND_PATH
from models.model_loader import load_models
from inference.inference_utils import run_hybrid
from utils.ip_location import get_location
from utils.weather_api import get_rainfall
from utils.logger import get_logger, log_to_csv

logger = get_logger("FloodCameraPro")

try:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
except:
    pygame = None

def play_alert():
    if pygame and os.path.exists(ALERT_SOUND_PATH):
        try: pygame.mixer.Sound(os.path.abspath(ALERT_SOUND_PATH)).play()
        except: pass

def start_camera():
    logger.info("Starting production-grade camera monitoring...")
    yolo, seg = load_models()
    
    # Context (Update once)
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Standardized Inference
        frame_out, water_p, objects, animals, risk = run_hybrid(frame, yolo, seg)
        
        # Logging (Phase 6)
        log_to_csv(water_p, objects, animals, rain, risk, city)
        
        # Alert (Phase 5)
        if risk in ["FLOOD", "DANGER"]:
            orig_h, orig_w = frame_out.shape[:2]
            cv2.putText(frame_out, f"{risk} ALERT", (orig_w//2-150, orig_h-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            play_alert()

        cv2.imshow("RAINWISE Pro Camera Feed", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
