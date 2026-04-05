import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from torchvision import transforms, models
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# PATH FIX
# -----------------------------
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DEVICE, NUM_CLASSES, MODEL_PATH, YOLO_MODEL_PATH, COLORS, INF_SIZE
from utils.area import calculate_water_area
from utils.risk import get_flood_risk
from utils.ip_location import get_location
from utils.weather_api import get_rainfall

app = FastAPI(title="RAINWISE Pro API")

YOLO_MODEL = None
SEG_MODEL = None

@app.on_event("startup")
async def load_models():
    global YOLO_MODEL, SEG_MODEL
    YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
    SEG_MODEL = models.segmentation.lraspp_mobilenet_v3_large(weights=None)
    SEG_MODEL.classifier.low_classifier = nn.Conv2d(40, NUM_CLASSES, 1)
    SEG_MODEL.classifier.high_classifier = nn.Conv2d(128, NUM_CLASSES, 1)
    if os.path.exists(MODEL_PATH):
        SEG_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    SEG_MODEL.to(DEVICE).eval()

def flood_logic_pro(img):
    """Standardized Hybrid Detection Logic for API"""
    orig_h, orig_w = img.shape[:2]
    
    # Context
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)
    
    # 1. YOLO (Primary)
    results = YOLO_MODEL(img, verbose=False)[0]
    obj_count = len(results.boxes)
    
    # 2. Segmentation (Secondary)
    with torch.no_grad():
        tensor = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize(INF_SIZE), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
        pred = torch.argmax(SEG_MODEL(tensor)["out"], dim=1)[0].cpu().numpy().astype(np.uint8)
    
    # 3. Standardized Analytics
    water_p = calculate_water_area(pred)
    risk = get_flood_risk(water_p, obj_count)
    
    return {
        "water_percent": round(water_p, 2),
        "severity": risk,
        "rainfall_mm": rain,
        "location": {
            "city": city,
            "lat": lat,
            "lon": lon
        },
        "detections": obj_count,
        "device": str(DEVICE),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/predict/pro")
async def predict_pro(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(status_code=400, detail="Invalid image")
    return JSONResponse(content=flood_logic_pro(img))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
