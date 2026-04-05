import os
import torch
from pathlib import Path

# -----------------------------
# ROOT & PATHS
# -----------------------------
# Current file is in project/src/config.py
# Root is two levels up from project/src/
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "project"

RAW_DIR = DATA_ROOT / "raw_dataset"
CLEAN_DIR = DATA_ROOT / "dataset_clean"
SPLIT_DIR = DATA_ROOT / "dataset_split"
WEIGHTS_DIR = DATA_ROOT / "weights"
OUTPUT_DIR = DATA_ROOT / "outputs"
LOGS_DIR = DATA_ROOT / "logs"
ASSETS_DIR = DATA_ROOT / "assets"

# -----------------------------
# MODELS
# -----------------------------
MODEL_PATH = WEIGHTS_DIR / "best_model.pth"
YOLO_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

# -----------------------------
# DEVICE (Auto-detection)
# -----------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# -----------------------------
# IMAGE & TRAINING SETTINGS
# -----------------------------
IMG_SIZE = (256, 256)   # Training size
INF_SIZE = (640, 640)   # Inference size for better quality
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

# -----------------------------
# CLASSES (Updated for Flood Detection Upgrade)
# -----------------------------
NUM_CLASSES = 6
CLASSES = ["flood_water", "road", "building", "vegetation", "vehicle", "animal"]

# 0: flood_water, 1: road, 2: building, 3: vegetation, 4: vehicle, 5: animal
COLORS = [
    (255, 0, 0),      # Blue (Flood Water)
    (0, 255, 0),      # Green (Road)
    (0, 0, 255),      # Red (Building)
    (0, 128, 255),    # Orange (Vegetation)
    (255, 255, 0),    # Cyan/Yellow (Vehicle)
    (255, 0, 255),    # Magenta (Animal)
]

# -----------------------------
# THRESHOLDS (PHASE 7)
# -----------------------------
YOLO_CONF_THRESHOLD = 0.3
WATER_LOW = 10
WATER_HIGH = 30
WATER_FLOOD = 50
ANIMAL_WATER_THRESHOLD = 20

# -----------------------------
# ALERTS & WEATHER
# -----------------------------
LAT = 22.3072
LON = 73.1812

EMAIL_USER = "your_email@gmail.com"
EMAIL_PASS = "your_app_password"
WEATHER_API_KEY = "your_openweathermap_api_key"

ALERT_SOUND_PATH = ASSETS_DIR / "alert.wav"
LOG_FILE_CSV = LOGS_DIR / "monitor_log.csv"
