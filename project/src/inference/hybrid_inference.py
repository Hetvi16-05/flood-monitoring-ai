import os
import sys
import cv2
import torch
from pathlib import Path

# -----------------------------
# PATH FIX (Standardized)
# -----------------------------
# File is in project/src/inference/hybrid_inference.py
project_root = Path(__file__).resolve().parents[3]
src_path = project_root / "project" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from config import OUTPUT_DIR, DATA_ROOT
from models.model_loader import load_models
from inference.inference_utils import run_hybrid
from utils.ip_location import get_location
from utils.weather_api import get_rainfall
from utils.logger import get_logger, log_to_csv

logger = get_logger("FloodBatchPro")

def process_batch():
    logger.info("Starting batch image inference...")
    yolo, seg = load_models()
    
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)
    
    test_dir = DATA_ROOT / "test_samples"
    out_dir = DATA_ROOT / "outputs" / "flood_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
        
    for img_name in [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png"))]:
        img_path = test_dir / img_name
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        
        # Standardized Inference
        frame_out, water_p, objects, animals, risk = run_hybrid(frame, yolo, seg)
        
        # Logging
        log_to_csv(water_p, objects, animals, rain, risk, city)
        
        save_path = out_dir / f"pro_{img_name}"
        cv2.imwrite(str(save_path), frame_out)
        logger.info(f"Processed: {img_name} -> {risk}")

if __name__ == "__main__":
    process_batch()
