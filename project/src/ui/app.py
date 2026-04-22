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

# --- INITIALIZE STATE IMMEDIATELY ---
st.set_page_config(page_title="RAINWISE Flood AI", page_icon="🌊", layout="wide")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if 'risk_history' not in st.session_state:
    st.session_state.risk_history = []
if 'conf_history' not in st.session_state:
    st.session_state.conf_history = []
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0
if 'last_telemetry' not in st.session_state:
    st.session_state.last_telemetry = None
if 'yt_video_path' not in st.session_state:
    st.session_state.yt_video_path = None

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

st.title("🌧️ RAINWISE Flood Monitoring AI")

@st.cache_resource
def get_cached_models():
    return load_models()

# --- MODEL ASSETS (Load early for sidebar) ---
model_bundle = get_cached_models()
models = model_bundle["models"]
metadata = model_bundle["metadata"]

# Extract metrics from log if available
latest_iou = 0.0
try:
    log_path = Path(__file__).resolve().parents[2] / "logs/training_log.csv"
    if log_path.exists():
        import pandas as pd
        df = pd.read_csv(log_path)
        latest_iou = df["Val_IoU"].max()
except:
    pass

# --- UTILS ---
def get_conf_color(conf):
    if conf > 0.8: return "🟢 High"
    if conf > 0.5: return "🟡 Medium"
    return "🔴 Low"

st.sidebar.divider()
st.sidebar.title("🧠 AI Model Identity")
st.sidebar.markdown(f"**Version:** `{metadata['seg_version']}`")
st.sidebar.markdown(f"**Status:** `🟢 {metadata['seg_status']}`")
st.sidebar.markdown(f"**Device:** `{metadata['device'].upper()}`")
st.sidebar.markdown(f"**Latest IoU:** `{latest_iou:.4f}`")

st.sidebar.divider()
st.sidebar.title("🎯 Operational Controls")
show_yolo = st.sidebar.toggle("Show YOLO Boxes", True)
show_mask = st.sidebar.toggle("Show Mask Overlay", True)
enable_sound = st.sidebar.toggle("Alert Audio", True)
enable_email = st.sidebar.toggle("Enable Email Alerts", False)
frame_skip = st.sidebar.slider(
    "FPS Optimization (Frame Skip)", 
    min_value=1, max_value=10, value=3, 
    help="Process 1 frame out of every N. Higher = Faster FPS"
)

st.sidebar.divider()
st.sidebar.title("🌍 Environmental Hub")

# Context Data
try:
    with st.spinner("Fetching Local Context..."):
        lat, lon, city = get_location()
        rain = get_rainfall(lat, lon)
    st.sidebar.metric("Location", city)
    st.sidebar.metric("Rainfall", f"{rain} mm")
    st.sidebar.map({"lat": [lat], "lon": [lon]}, zoom=10)
except Exception as e:
    city = "Unknown"
    rain = 0
    st.sidebar.error(f"Context Error: {e}")


def process_and_display(frame, main_container, rek_container, mode="BATCH", lat=None, lon=None):
    """Production-grade Rekognition-style Monitoring Logic"""
    res_img, water_p, obj_summary, risk_level, risk_score, telemetry = run_hybrid(frame, models, show_yolo, show_mask, lat, lon)
    
    hybrid_conf = telemetry['hybrid_conf']
    st.session_state.last_frame = frame.copy()
    st.session_state.last_telemetry = telemetry
    st.session_state.last_metadata = {
        "water_p": water_p, "obj_summary": obj_summary, 
        "risk_level": risk_level, "risk_score": risk_score, 
        "telemetry": telemetry, "mode": mode
    }

    # 1. Main Display
    main_container.image(res_img, channels="BGR", use_container_width=True)
    
    # 2. Alert Banner System
    if risk_level in ["HIGH", "DANGEROUS"] and hybrid_conf > 0.75:
        main_container.error(f"🚨 **CRITICAL FLOOD ALERT:** {risk_level} Risk Detected ({risk_score}/100) with {hybrid_conf:.1%} Confidence!")
        if enable_sound and pygame and os.path.exists(ALERT_SOUND_PATH):
            try: pygame.mixer.Sound(os.path.abspath(ALERT_SOUND_PATH)).play()
            except: pass
    
    # 3. Rekognition Side Panel
    with rek_container.container():
        st.markdown("### 🎯 Real-Time Analysis")
        
        # Hybrid Confidence Card
        conf_status = get_conf_color(hybrid_conf)
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid {'#00ff00' if hybrid_conf > 0.8 else '#ffff00' if hybrid_conf > 0.5 else '#ff4b4b'}">
            <h4 style="margin:0; color:white;">Hybrid Confidence</h4>
            <h2 style="margin:0; color:white;">{hybrid_conf:.1%}</h2>
            <p style="margin:0; color:#aaa;">Status: {conf_status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Water Depth Information (NEW)
        if telemetry.get('avg_water_depth', 0) > 0:
            st.markdown("#### 📏 Water Depth")
            c1, c2 = st.columns([1, 1])
            c1.metric("Avg Depth", f"{telemetry['avg_water_depth']:.2f}m")
            c2.metric("Max Depth", f"{telemetry['max_water_depth']:.2f}m")
            if telemetry.get('depth_risk_level'):
                depth_color = '🔴' if telemetry['depth_risk_level'] == 'DANGEROUS' else '🟠' if telemetry['depth_risk_level'] == 'HIGH' else '🟡' if telemetry['depth_risk_level'] == 'MEDIUM' else '🟢'
                st.markdown(f"Depth Risk: {depth_color} **{telemetry['depth_risk_level']}**")
        
        st.divider()
        
        # Temporal Flood Tracking Information (NEW)
        if telemetry.get('expansion_rate', 0) != 0:
            st.markdown("#### ⏱️ Flood Progression")
            c1, c2 = st.columns([1, 1])
            c1.metric("Expansion Rate", f"{telemetry['expansion_rate']*100:.1f}%")
            c2.metric("Direction", telemetry['flood_direction'].upper())
            if telemetry.get('temporal_risk_level'):
                temporal_color = '🔴' if telemetry['temporal_risk_level'] == 'DANGEROUS' else '🟠' if telemetry['temporal_risk_level'] == 'HIGH' else '🟡' if telemetry['temporal_risk_level'] == 'MEDIUM' else '🟢'
                st.markdown(f"Temporal Risk: {temporal_color} **{telemetry['temporal_risk_level']}**")
            if telemetry.get('temporal_insights'):
                st.markdown("**Insights:**")
                for insight in telemetry['temporal_insights']:
                    st.markdown(f"• {insight}")
        
        st.divider()
        
        # Predictive Forecasting Information (NEW)
        if telemetry.get('predictive_risk_level'):
            st.markdown("#### 🔮 Flood Prediction")
            c1, c2 = st.columns([1, 1])
            c1.metric("Risk Level", telemetry['predictive_risk_level'])
            c2.metric("Hours Ahead", f"{telemetry['prediction_hours']}h")
            pred_color = '🔴' if telemetry['predictive_risk_level'] == 'DANGEROUS' else '🟠' if telemetry['predictive_risk_level'] == 'HIGH' else '🟡' if telemetry['predictive_risk_level'] == 'MEDIUM' else '🟢'
            st.markdown(f"Prediction: {pred_color} **{telemetry['predictive_risk_level']}**")
            if telemetry.get('prediction_factors'):
                st.markdown("**Factors:**")
                for factor in telemetry['prediction_factors']:
                    st.markdown(f"• {factor}")
        
        st.divider()
        # Detected Labels section removed per user request
        # st.markdown("#### 🏷️ Detected Labels")
        # for det in telemetry['top_detections'][:5]: # Show top 5
        #     c1, c2 = st.columns([3, 1])
        #     c1.markdown(f"**{det['label']}**")
        #     c2.markdown(f"`{det['confidence']:.1%}`")
        #     st.progress(det['confidence'])

        st.divider()
        st.markdown("#### 📈 Confidence Trend")
        st.session_state.conf_history.append(hybrid_conf)
        if len(st.session_state.conf_history) > 50: st.session_state.conf_history.pop(0)
        
        # Detect instability
        if len(st.session_state.conf_history) > 10:
            recent_mean = np.mean(st.session_state.conf_history[-10:])
            if abs(hybrid_conf - recent_mean) > 0.15:
                st.warning("⚠️ Unstable prediction detected")
        
        st.line_chart(st.session_state.conf_history, height=120)
        
        st.markdown("#### 🛡️ Risk Engine")
        st.metric("Risk Level", risk_level, delta=risk_score, delta_color="inverse")
        st.session_state.risk_history.append(risk_score)
        if len(st.session_state.risk_history) > 50: st.session_state.risk_history.pop(0)
        st.line_chart(st.session_state.risk_history, height=120)

    # 4. Advanced CSV Logging
    log_to_csv(water_p, obj_summary, risk_level, risk_score, rain, city)
    return risk_level

def handle_feedback(is_good, reason="N/A"):
    if 'last_frame' not in st.session_state:
        st.error("No frame to report!")
        return
    
    frame = st.session_state.last_frame
    meta = st.session_state.last_metadata
    meta['user_feedback'] = "correct" if is_good else "incorrect"
    meta['failure_reason'] = reason
    meta['timestamp'] = datetime.now().isoformat()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "REWARD" if is_good else "RETRAIN"
    
    os.makedirs(ROOT / "feedback/images", exist_ok=True)
    os.makedirs(ROOT / "feedback/metadata", exist_ok=True)
    
    img_path = f"feedback/images/{label}_{ts}.jpg"
    meta_path = f"feedback/metadata/{label}_{ts}.json"
    
    cv2.imwrite(str(ROOT / img_path), frame)
    import json
    with open(ROOT / meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    if is_good: st.success("✅ Feedback Recorded! Model rewarded.")
    else: st.warning(f"🛠️ Reported for Retraining! Reason: {reason}")

# --- TABS ---
tab_img, tab_vid, tab_cctv = st.tabs(["📸 Image", "🎥 Video", "📡 CCTV Live"])

# --- IMAGE TAB ---
with tab_img:
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="img_up")
    if img_file:
        image = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        col_main, col_rek = st.columns([3, 1])
        try:
            process_and_display(frame, col_main, col_rek, mode="IMAGE", lat=lat, lon=lon)
        except NameError:
            # lat/lon not available (location fetch failed)
            process_and_display(frame, col_main, col_rek, mode="IMAGE")
        
        st.divider()
        st.subheader("🗣️ Active Learning Feedback")
        f1, f2, f3 = st.columns([1,1,2])
        reason = f3.selectbox("Failure Reason (if applicable)", 
                            ["N/A", "Wrong flood detection", "Missed object", "False positive", "Low confidence"])
        if f1.button("🎖️ Reward Model", use_container_width=True):
            handle_feedback(True)
        if f2.button("🛠️ Send to Retrain", use_container_width=True):
            handle_feedback(False, reason)

# --- VIDEO TAB ---
with tab_vid:
    st.subheader("🎥 Intelligent Video Analysis")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        vid_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="vid_up")
        
    with c2:
        st.markdown("🔗 **Cloud Import**")
        yt_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        if st.button("📥 Download & Process"):
            if yt_url:
                with st.spinner("📦 Downloading YouTube Stream..."):
                    import yt_dlp
                    ydl_opts = {
                        'format': 'best[ext=mp4]',
                        'outtmpl': 'project/test_videos/%(title)s.%(ext)s',
                        'noplaylist': True,
                    }
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(yt_url, download=True)
                            yt_path = ydl.prepare_filename(info)
                            st.success(f"✅ Downloaded: {info['title']}")
                            st.session_state.yt_video_path = yt_path
                    except Exception as e:
                        st.error(f"Download Error: {e}")
            else:
                st.warning("Please enter a valid URL")

    # Determine which video to use
    active_vid_path = None
    if vid_file:
        tmp_path = "tmp_video.mp4"
        with open(tmp_path, "wb") as f: f.write(vid_file.read())
        active_vid_path = tmp_path
    elif 'yt_video_path' in st.session_state:
        if st.session_state.yt_video_path and os.path.exists(st.session_state.yt_video_path):
            active_vid_path = st.session_state.yt_video_path
            st.info(f"Using Cloud Video: {os.path.basename(active_vid_path)}")

    if active_vid_path:
        cap = cv2.VideoCapture(active_vid_path)
        
        col_main, col_rek = st.columns([3, 1])
        v_main = col_main.empty()
        v_rek = col_rek.empty()
        
        if st.button("🚀 Start Analysis"):
            st.session_state.risk_history = []
            st.session_state.conf_history = []
            f_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if f_count % frame_skip == 0:
                    try:
                        process_and_display(frame, v_main, v_rek, mode="VIDEO", lat=lat, lon=lon)
                    except NameError:
                        process_and_display(frame, v_main, v_rek, mode="VIDEO")
                f_count += 1
            cap.release()
        
        st.divider()
        st.subheader("🗣️ Video Feedback (Last Evaluated Frame)")
        f1, f2, f3 = st.columns([1,1,2])
        v_reason = f3.selectbox("Failure Reason", 
                              ["N/A", "Wrong flood detection", "Missed object", "False positive", "Low confidence"], key="v_reason")
        if f1.button("🎖️ Reward", use_container_width=True, key="v_rew"):
            handle_feedback(True)
        if f2.button("🛠️ Retrain", use_container_width=True, key="v_ret"):
            handle_feedback(False, v_reason)

# --- CCTV LIVE TAB ---
with tab_cctv:
    st.subheader("📡 Live Flood Surveillance")
    cctv_url = st.text_input("Stream URL (0 for Webcam)", value="0")
    
    col_main, col_rek = st.columns([3, 1])
    c_main = col_main.empty()
    c_rek = col_rek.empty()
    
    c1, c2 = st.columns(2)
    start_btn = c1.button("🔴 Start Live Stream", use_container_width=True)
    stop_btn = c2.button("⏹️ Stop Stream", use_container_width=True)
    
    if start_btn:
        try:
            url_val = int(cctv_url) if cctv_url.isdigit() else cctv_url
            cap = cv2.VideoCapture(url_val)
            f_count = 0
            st.session_state.risk_history = []
            st.session_state.conf_history = []
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret: break
                if f_count % frame_skip == 0:
                    try:
                        process_and_display(frame, c_main, c_rek, mode="CCTV LIVE", lat=lat, lon=lon)
                    except NameError:
                        process_and_display(frame, c_main, c_rek, mode="CCTV LIVE")
                f_count += 1
            cap.release()
        except Exception as e:
            st.error(f"Stream failure: {e}")
    
    st.divider()
    st.subheader("🗣️ Live Feedback")
    f1, f2, f3 = st.columns([1,1,2])
    c_reason = f3.selectbox("Failure Reason", 
                          ["N/A", "Wrong flood detection", "Missed object", "False positive", "Low confidence"], key="c_reason")
    if f1.button("🎖️ Reward", use_container_width=True, key="c_rew"):
        handle_feedback(True)
    if f2.button("🛠️ Retrain", use_container_width=True, key="c_ret"):
        handle_feedback(False, c_reason)

st.divider()
st.caption("🔒 RAINWISE Enterprise AI | Secure, Real-Time Flood Intelligence | Optimized for Apple Silicon")
