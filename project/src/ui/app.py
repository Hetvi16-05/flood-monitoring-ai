import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path
from PIL import Image
from datetime import datetime

# -----------------------------
# PATH FIXES
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]   # project/
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
from utils.email_alert import send_flood_alert

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
# STATE INITIALIZATION
# -----------------------------
if 'risk_history' not in st.session_state:
    st.session_state.risk_history = []
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0

# -----------------------------
# UI CONTROLS (Sidebar)
# -----------------------------
st.sidebar.title("Operational Controls")
show_yolo = st.sidebar.toggle("Show YOLO Boxes", True)
show_mask = st.sidebar.toggle("Show Mask Overlay", True)
enable_sound = st.sidebar.toggle("Alert Audio", True)
enable_email = st.sidebar.toggle("Enable Email Alerts", False)
frame_skip = st.sidebar.slider(
    "FPS Optimization (Frame Skip)", 
    min_value=1, max_value=10, value=3, 
    help="Process 1 frame out of every N for Video/CCTV. Higher = Faster FPS"
)

st.sidebar.divider()
st.sidebar.title("Environmental Hub")

# Context Data
try:
    with st.spinner("Fetching Local Context..."):
        lat, lon, city = get_location()
        rain = get_rainfall(lat, lon)
    st.sidebar.metric("Location", city)
    st.sidebar.metric("Rainfall", f"{rain} mm")
    st.sidebar.map({"lat": [lat], "lon": [lon]})
except Exception as e:
    city = "Unknown"
    rain = 0
    st.sidebar.error(f"Context Error: {e}")

# -----------------------------
# MAIN UI
# -----------------------------
models = get_cached_models()

tab_img, tab_vid, tab_cctv = st.tabs(["📸 Image Monitoring", "🎥 Video Monitoring", "📡 CCTV Live Alert"])

def process_and_display(frame, container, metrics_container, chart_container, mode="BATCH"):
    """Enhanced processing and display logic for production hybrid system"""
    res_img, water_p, obj_summary, risk_level, risk_score, confidence = run_hybrid(frame, models, show_yolo, show_mask)
    
    # Store for feedback
    st.session_state.last_frame = frame.copy()
    st.session_state.last_metadata = {
        "water_p": water_p, "obj_summary": obj_summary, 
        "risk_level": risk_level, "risk_score": risk_score, 
        "confidence": confidence, "mode": mode
    }

    # 1. Render Processed Frame
    container.image(res_img, channels="BGR", use_container_width=True)
    
    # 2. Enhanced UI Metrics Dashboard
    with metrics_container.container():
        st.markdown(f"### 🛡️ Real-Time Intelligence (`{mode}`)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Level", risk_level)
        c2.metric("Risk Score", f"{risk_score}/100")
        c3.metric("Water Coverage", f"{water_p:.1f}%")
        c4.metric("AI Confidence", f"{confidence:.1%}")
        st.markdown(f"**Spatial Targets:** `{obj_summary}`")
    
    # 3. Analytics Chart
    st.session_state.risk_history.append(risk_score)
    if len(st.session_state.risk_history) > 60:  # Keep timeline to last 60 evaluated chunks
        st.session_state.risk_history.pop(0)
        
    if chart_container is not None:
        with chart_container.container():
            st.markdown("##### 📈 Risk Score Trend")
            st.line_chart(st.session_state.risk_history, height=150)

    # 4. Debounced Alert System
    if risk_level in ["HIGH", "DANGEROUS"]:
        st.toast(f"🚨 {risk_level} FLOODING (Score: {risk_score})", icon="⚠️")
        
        # Audio
        if enable_sound and pygame and os.path.exists(ALERT_SOUND_PATH):
            try: pygame.mixer.Sound(os.path.abspath(ALERT_SOUND_PATH)).play()
            except: pass
            
        # Email Dispatch (Handled natively by email_alert.py)
        if enable_email:
            send_flood_alert(risk_level, risk_score, water_p, obj_summary, city)

    # 5. Advanced CSV Logging
    log_to_csv(water_p, obj_summary, risk_level, risk_score, rain, city)
    return risk_level

def handle_feedback(is_good):
    if 'last_frame' not in st.session_state:
        st.error("No frame to report!")
        return
    
    frame = st.session_state.last_frame
    meta = st.session_state.last_metadata
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "REWARD" if is_good else "RETRAIN"
    
    img_path = f"feedback/images/{label}_{ts}.jpg"
    meta_path = f"feedback/metadata/{label}_{ts}.json"
    
    cv2.imwrite(str(ROOT / img_path), frame)
    import json
    with open(ROOT / meta_path, 'w') as f:
        json.dump(meta, f)
    
    if is_good:
        st.success(f"✅ Feedback Recorded! Model rewarded for correct detection.")
    else:
        st.warning(f"🛠️ Reported! Image saved to feedback/ for model retraining.")

# --- IMAGE TAB ---
with tab_img:
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        m_col = st.empty()
        c_col = st.empty()
        process_and_display(frame, st.empty(), m_col, c_col, mode="IMAGE")
        
        st.divider()
        st.subheader("🗣️ Human-in-the-Loop Feedback")
        f1, f2 = st.columns(2)
        if f1.button("🎖️ Reward Model (Correct)", key="img_reward"):
            handle_feedback(True)
        if f2.button("🛠️ Send to Retrain (Error)", key="img_retrain"):
            handle_feedback(False)

# --- VIDEO TAB ---
with tab_vid:
    vid_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if vid_file:
        tmp_path = "tmp_video.mp4"
        with open(tmp_path, "wb") as f: f.write(vid_file.read())
        cap = cv2.VideoCapture(tmp_path)
        
        col1, col2 = st.columns([2, 1])
        v_placeholder = col1.empty()
        v_metrics = col2.empty()
        v_chart = col2.empty()
        
        if st.button("Start Analysis"):
            st.session_state.risk_history = []
            f_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Inference Optimization
                if f_count % frame_skip == 0:
                    process_and_display(frame, v_placeholder, v_metrics, v_chart, mode="VIDEO")
                f_count += 1
            cap.release()
        
        st.divider()
        st.subheader("🗣️ Feedback on Last Analyzed Frame")
        f1, f2 = st.columns(2)
        if f1.button("🎖️ Reward Model", key="vid_reward"):
            handle_feedback(True)
        if f2.button("🛠️ Report Error", key="vid_retrain"):
            handle_feedback(False)

# --- CCTV LIVE TAB ---
with tab_cctv:
    st.subheader("📡 Live Flood Surveillance")
    cctv_url = st.text_input("Stream URL (0 for Webcam)", value="0")
    
    col1, col2 = st.columns([2, 1])
    cctv_placeholder = col1.empty()
    cctv_metrics = col2.empty()
    cctv_chart = col2.empty()
    
    start_btn = st.button("🔴 Start Live Stream")
    stop_btn = st.button("⏹️ Stop Stream")
    
    if start_btn:
        try:
            url_val = int(cctv_url) if cctv_url.isdigit() else cctv_url
            cap = cv2.VideoCapture(url_val)
            f_count = 0
            st.session_state.risk_history = []
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret: break
                
                if f_count % frame_skip == 0:
                    process_and_display(frame, cctv_placeholder, cctv_metrics, cctv_chart, mode="CCTV LIVE")
                f_count += 1
            cap.release()
        except Exception as e:
            st.error(f"Stream failure: {e}")
    
    st.divider()
    st.subheader("🗣️ Live Feedback (Last Frame)")
    f1, f2 = st.columns(2)
    if f1.button("🎖️ Reward", key="cctv_reward"):
        handle_feedback(True)
    if f2.button("🛠️ Retrain", key="cctv_retrain"):
        handle_feedback(False)

st.divider()
st.caption("🔒 RAINWISE Enterprise AI | Secure, Real-Time Flood Intelligence")
