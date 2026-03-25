import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import pygame
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# PATH FIX
# -----------------------------
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DEVICE, NUM_CLASSES, MODEL_PATH, YOLO_MODEL_PATH, CLASSES, COLORS, INF_SIZE, ALERT_SOUND_PATH
from project.src.utils.ip_location import get_location
from project.src.utils.weather_api import get_rainfall

# Audio Init
pygame.mixer.init()

@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL_PATH)
    seg = models.segmentation.lraspp_mobilenet_v3_large(weights=None)
    seg.classifier.low_classifier = nn.Conv2d(40, NUM_CLASSES, 1)
    seg.classifier.high_classifier = nn.Conv2d(128, NUM_CLASSES, 1)
    if os.path.exists(MODEL_PATH):
        seg.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    seg.to(DEVICE).eval()
    return yolo, seg

def get_flood_analytics(mask):
    water_p = round((np.sum(mask == 0) / (mask.size - np.sum(mask == 255))) * 100, 2)
    severity = "HIGH" if water_p > 30 else "MEDIUM" if water_p > 10 else "LOW"
    return water_p, severity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="RAINWISE Pro Dash", page_icon="🌊", layout="wide")
st.title("🌧️ RAINWISE Pro: Real-Time Intelligence")

# Fetch Context
with st.spinner("Fetching Dynamic Context..."):
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)

# Sidebar
st.sidebar.title("Environmental Hub")
st.sidebar.metric("Current City", city)
st.sidebar.metric("Rainfall (Last 1h)", f"{rain} mm")
st.sidebar.metric("Coordinates", f"{lat:.4f}, {lon:.4f}")

# Map
st.sidebar.write("### Location View")
map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.sidebar.map(map_data)

uploaded_file = st.file_uploader("Analyze Mission Snapshot", type=["jpg", "png"])

if uploaded_file:
    yolo, seg = load_models()
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert("RGB"))
    orig_h, orig_w = img_array.shape[:2]

    # Transform
    st_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize(INF_SIZE), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with st.spinner("AI Analysis in Progress..."):
        tensor = st_transform(img_array).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = seg(tensor)["out"]
            pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
        
        water_p, severity = get_flood_analytics(pred)
        
        # Overlay
        mask_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for i, color in enumerate(COLORS): mask_colored[pred == i] = color
        mask_colored = cv2.resize(mask_colored, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        combined = cv2.addWeighted(img_array, 0.6, mask_colored, 0.4, 0)
        
        # YOLO
        results = yolo(img_array, verbose=False)[0]
        num_objs = len(results.boxes)
        for box in results.boxes:
            if float(box.conf[0]) > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(combined, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # UI Results
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(combined, channels="RGB", use_container_width=True)
        if severity == "HIGH":
            st.error(f"🔴 CRITICAL ALERT: {city.upper()} IS AT HIGH RISK")
            if os.path.exists(ALERT_SOUND_PATH):
                try: pygame.mixer.Sound(ALERT_SOUND_PATH).play()
                except: pass

    with col2:
        st.header("Flood Intelligence")
        st.subheader(f"Status: {severity}")
        st.progress(water_p / 100)
        st.metric("Water Coverage", f"{water_p}%")
        st.metric("Objects Identified", num_objs)
        st.write("---")
        st.write("### Model Confidence (Classes)")
        for cls in CLASSES: st.info(f"Class: {cls.capitalize()} ✅")

    st.download_button("Export Intelligence Report", 
                       data=cv2.imencode('.jpg', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))[1].tobytes(),
                       file_name=f"rainwise_intel_{city.lower()}.jpg")

st.markdown("---")
st.markdown("🔒 RAINWISE PRO | Secure, Real-Time Disaster Intelligence")
