import os
import sys
import cv2
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# PATH FIX (Standardized)
# -----------------------------
# File is in project/src/inference/hybrid_video.py
project_root = Path(__file__).resolve().parents[3]
src_path = project_root / "project" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from config import ALERT_SOUND_PATH, OUTPUT_DIR
from models.model_loader import load_models
from inference.inference_utils import run_hybrid
from utils.ip_location import get_location
from utils.weather_api import get_rainfall
from utils.logger import get_logger, log_to_csv

logger = get_logger("FloodVideoPro")

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

def process_video(input_path):
    if not os.path.exists(input_path):
        logger.error(f"Video file not found: {input_path}")
        return

    logger.info(f"Processing video: {input_path}")
    yolo, seg = load_models()
    
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)
    
    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps, total = cap.get(5), int(cap.get(7))

    out_dir = Path(OUTPUT_DIR) / "video_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"pro_{os.path.basename(input_path)}"
    
    out_video = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    pbar = tqdm(total=total, desc="Pro Video Analysis")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Standardized Inference
        frame_out, water_p, objects, animals, risk = run_hybrid(frame, yolo, seg)
        
        # Logging
        log_to_csv(water_p, objects, animals, rain, risk, city)
        
        # Alert
        if risk in ["FLOOD", "DANGER"]:
            cv2.putText(frame_out, f"{risk} ALERT", (width//2-150, height-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            play_alert()

        out_video.write(frame_out)
        pbar.update(1)

    cap.release()
    out_video.release()
    pbar.close()
    logger.info(f"Video saved to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_video(sys.argv[1])
    else:
        print("Usage: python hybrid_video.py <video_path>")
