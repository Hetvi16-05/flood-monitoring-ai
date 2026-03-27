import os
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
from .config import DEVICE, NUM_CLASSES, MODEL_PATH, YOLO_MODEL_PATH

def load_models():
    """
    Load YOLO and Segmentation models with error handling.
    """
    # 1. Load YOLOv8
    yolo = YOLO(YOLO_MODEL_PATH)
    
    # 2. Load Custom LRASPP
    seg = models.segmentation.lraspp_mobilenet_v3_large(weights=None)
    seg.classifier.low_classifier = nn.Conv2d(40, NUM_CLASSES, 1)
    seg.classifier.high_classifier = nn.Conv2d(128, NUM_CLASSES, 1)
    
    if os.path.exists(MODEL_PATH):
        seg.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    seg.to(DEVICE).eval()
    return yolo, seg
