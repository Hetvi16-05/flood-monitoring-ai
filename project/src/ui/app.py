import streamlit as st
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image
from datetime import datetime

# -----------------------------
# PATH FIX (Standardized)
# -----------------------------
# File is in project/src/ui/app.py
project_root = Path(__file__).resolve().parents[3]
src_path = project_root / "project" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from config import (
    YOLO_MODEL_PATH, MODEL_PATH, DEVICE, ALERT_SOUND_PATH,
    LAT, LON
)
from models.model_loader import load_models
from inference.inference_utils import run_hybrid
from inference.cctv_stream import CCTVStream
from utils.ip_location import get_location
from utils.weather_api import get_rainfall
from utils.logger import log_to_csv

# Audio Init safely
try:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
except:
    pygame = None

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="RAINWISE Flood AI", page_icon="🌊", layout="wide")
st.title("🌧️ RAINWISE Flood Monitoring AI")

@st.cache_resource
def get_cached_models():
    return load_models()

# -----------------------------
# UI CONTROLS (Sidebar)
# -----------------------------
st.sidebar.title("Operational Controls")
show_yolo = st.sidebar.toggle("Show YOLO Boxes", True)
show_mask = st.sidebar.toggle("Show Mask Overlay", True)
enable_sound = st.sidebar.toggle("Alert Audio", True)

st.sidebar.divider()
st.sidebar.title("Environmental Hub")

# Context Data
with st.spinner("Fetching Local Context..."):
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)

st.sidebar.metric("Location", city)
st.sidebar.metric("Rainfall", f"{rain} mm")
st.sidebar.map({"lat": [lat], "lon": [lon]})

# -----------------------------
# MAIN UI
# -----------------------------
yolo, seg = get_cached_models()

tab_img, tab_vid, tab_cam, tab_cctv = st.tabs(["📸 Image Upload", "🎥 Video Upload", "📹 Live Camera", "📡 CCTV/RTSP"])

def process_and_display(frame, container, metrics_container):
    """Common processing and display logic"""
    res_img, water_p, objects, animals, risk = run_hybrid(frame, yolo, seg, show_yolo, show_mask)
    
    # Display result
    container.image(res_img, channels="BGR", use_container_width=True)
    
    # Update metrics
    metrics_container.markdown(f"""
    ### Detection Metrics
    - **Risk Level:** `{risk}`
    - **Water Coverage:** `{water_p:.1f}%`
    - **Objects:** `{", ".join(objects) if objects else "None"}`
    - **Animals:** `{", ".join(animals) if animals else "None"}`
    - **Rainfall:** `{rain} mm`
    - **Location:** `{city}`
    """)
    
    if risk in ["FLOOD", "DANGER"]:
        st.toast(f"🚨 {risk} DETECTED!", icon="⚠️")
        if enable_sound and pygame and os.path.exists(ALERT_SOUND_PATH):
            try: pygame.mixer.Sound(os.path.abspath(ALERT_SOUND_PATH)).play()
            except: pass
            
    # Always log
    log_to_csv(water_p, objects, animals, rain, risk, city)
    return risk

# --- IMAGE TAB ---
with tab_img:
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            placeholder = st.empty()
        with col2:
            metrics_placeholder = st.empty()
            
        with st.spinner("Analyzing..."):
            process_and_display(frame, placeholder, metrics_placeholder)

# --- VIDEO TAB ---
with tab_vid:
    vid_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if vid_file:
        tmp_path = "tmp_video.mp4"
        with open(tmp_path, "wb") as f: f.write(vid_file.read())
        
        cap = cv2.VideoCapture(tmp_path)
        col1, col2 = st.columns([3, 1])
        v_placeholder = col1.empty()
        v_metrics = col2.empty()
        
        if st.button("Start Processing"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                process_and_display(frame, v_placeholder, v_metrics)
            cap.release()

# --- CAMERA TAB ---
with tab_cam:
    if st.button("Activate Live Monitoring"):
        cap = cv2.VideoCapture(0)
        col1, col2 = st.columns([3, 1])
        c_placeholder = col1.empty()
        c_metrics = col2.empty()
        
        stop = st.button("Emergency Stop", key="cam_stop")
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret: break
            process_and_display(frame, c_placeholder, c_metrics)
        cap.release()

# --- CCTV TAB ---
with tab_cctv:
    st.subheader("📡 CCTV / RTSP Monitoring")
    cctv_url = st.text_input("Enter Stream URL", placeholder="rtsp://user:pass@ip:554/stream1", help="Supports RTSP, HTTP, and IP Camera streams.")
    
    col1, col2 = st.columns([3, 1])
    cctv_placeholder = col1.empty()
    cctv_metrics = col2.empty()
    
    if st.button("Start CCTV Feed"):
        if not cctv_url:
            st.warning("Please enter a valid URL.")
        else:
            try:
                stream = CCTVStream(cctv_url)
                stop_cctv = st.button("Stop Monitoring", key="cctv_stop")
                while not stop_cctv:
                    # CCTVStream handles run_hybrid internally or we can do it here for consistency
                    # Let's read the frame and process it using our common function
                    ret, raw_frame = stream.cap.read()
                    if not ret:
                        st.error("Connection lost or stream ended.")
                        break
                    
                    process_and_display(raw_frame, cctv_placeholder, cctv_metrics)
                stream.release()
            except Exception as e:
                st.error(f"Stream Error: {e}")

st.divider()
st.caption("🔒 RAINWISE Enterprise AI | Secure, Real-Time Flood Intelligence")
