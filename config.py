import os
import torch

# -----------------------------
# ROOT
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "project")

# -----------------------------
# PATHS
# -----------------------------
RAW_DIR = os.path.join(DATA_ROOT, "raw_dataset")
CLEAN_DIR = os.path.join(DATA_ROOT, "dataset_clean")
SPLIT_DIR = os.path.join(DATA_ROOT, "dataset_split")
WEIGHTS_DIR = os.path.join(DATA_ROOT, "weights")
OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs")
LOGS_DIR = os.path.join(DATA_ROOT, "logs")

# -----------------------------
# MODELS
# -----------------------------
# Custom LRASPP segmentation model
MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.pth")

# Pre-trained YOLOv8 detection model
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

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
# IMAGE SETTINGS
# -----------------------------
IMG_SIZE = (256, 256)   # Training size
INF_SIZE = (640, 640)   # Inference size for better quality

# -----------------------------
# CLASSES
# -----------------------------
NUM_CLASSES = 4
CLASSES = ["water", "road", "building", "vegetation"]

# -----------------------------
# TRAIN SETTINGS
# -----------------------------
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4

# 0: water, 1: road, 2: building, 3: vegetation
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (0, 128, 255),    # Orange
]

# -----------------------------
# PRO UPGRADE: ALERTS & WEATHER
# -----------------------------
# Default coordinates (will be overridden by IP location)
LAT = 22.3072
LON = 73.1812

EMAIL_USER = "your_email@gmail.com"
EMAIL_PASS = "your_app_password"

WEATHER_API_KEY = "your_openweathermap_api_key"

ALERT_SOUND_PATH = os.path.join(PROJECT_ROOT, "project", "assets", "alert.wav")