import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
from PIL import Image
from datetime import datetime

# -----------------------------
# PATH FIXES
# -----------------------------
ROOT = Path(__file__).resolve().parents[3]   # Flood_Prediction/
SRC = ROOT / "project" / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (
    YOLO_MODEL_PATH, MODEL_PATH, DEVICE, ALERT_SOUND_PATH,
    LAT, LON
)
from models.model_loader import load_models
from inference.inference_utils import run_hybrid
from inference.advanced_inference import AdvancedHybridInference, run_advanced_hybrid
from utils.ip_location import get_location
from utils.weather_api import get_rainfall
from utils.logger import log_to_csv

app = Flask(__name__)

# Global state
advanced_engine = None
last_telemetry = {}
risk_history = []
conf_history = []
user_settings = {
    "show_yolo": True,
    "show_mask": True,
    "enable_sound": True,
    "enable_email": False,
    "alert_email": ""
}

def sanitize_data(data):
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(i) for i in data]
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    return data

def get_advanced_engine():
    global advanced_engine
    if advanced_engine is None:
        print("🛠️ Initializing Advanced Custom AI Engine...")
        advanced_engine = AdvancedHybridInference()
    return advanced_engine

@app.route('/')
def index():
    lat, lon, city = get_location()
    rain = get_rainfall(lat, lon)
    return render_template('index.html', city=city, rain=rain, lat=lat, lon=lon)

@app.route('/status')
def status():
    global last_telemetry, risk_history, conf_history
    return jsonify(sanitize_data({
        "telemetry": last_telemetry,
        "risk_history": risk_history,
        "conf_history": conf_history
    }))

def gen_frames(camera_source=0):
    global last_telemetry, risk_history, conf_history
    cap = cv2.VideoCapture(camera_source)
    engine = get_advanced_engine()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process frame using Custom Advanced Architecture
            res_img, water_p, obj_summary, risk_level, risk_score, telemetry = run_advanced_hybrid(
                frame, engine
            )
            
            # Update global state for /status endpoint
            last_telemetry = telemetry
            last_telemetry['risk_level'] = risk_level
            last_telemetry['risk_score'] = risk_score
            last_telemetry['water_p'] = water_p
            
            risk_history.append(risk_score)
            if len(risk_history) > 50: risk_history.pop(0)
            
            conf_history.append(telemetry['hybrid_conf'])
            if len(conf_history) > 50: conf_history.pop(0)

            # Log to CSV
            lat, lon, city = get_location()
            rain = get_rainfall(lat, lon)
            log_to_csv(water_p, obj_summary, risk_level, risk_score, rain, city)

            # Encode and yield
            ret, buffer = cv2.imencode('.jpg', res_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    source = request.args.get('source', 0)
    try:
        source = int(source)
    except:
        pass
    return Response(gen_frames(source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    global last_telemetry, risk_history, conf_history
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        engine = get_advanced_engine()
        
        res_img, water_p, obj_summary, risk_level, risk_score, telemetry = run_advanced_hybrid(
            frame, engine
        )
        
        # Update state
        last_telemetry = telemetry
        last_telemetry['risk_level'] = str(risk_level)
        last_telemetry['risk_score'] = float(risk_score)
        last_telemetry['water_p'] = float(water_p)
        
        # Log to CSV
        lat, lon, city = get_location()
        rain = get_rainfall(lat, lon)
        log_to_csv(water_p, obj_summary, risk_level, risk_score, rain, city)
        
        # Encode result image
        _, buffer = cv2.imencode('.jpg', res_img)
        import base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify(sanitize_data({
            "result_image": img_base64,
            "telemetry": last_telemetry,
            "risk_level": str(risk_level),
            "risk_score": float(risk_score),
            "water_p": float(water_p)
        }))
    except Exception as e:
        import traceback
        print(f"❌ Error in /predict: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save video to tmp location
    ext = os.path.splitext(file.filename)[1]
    tmp_name = f"uploaded_video{ext}"
    path = os.path.join(ROOT, "project", "test_videos", tmp_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file.save(path)
    
    return jsonify({"success": True, "path": path})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global user_settings
    data = request.json
    user_settings.update(data)
    return jsonify({"success": True, "settings": user_settings})

@app.route('/process_youtube', methods=['POST'])
def process_youtube():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    # We will use yt-dlp to download
    # For now, we'll just mock the path or use a temporary one
    # In a real app, you'd want to handle this asynchronously
    try:
        import yt_dlp
        save_dir = os.path.join(ROOT, "project", "test_videos")
        os.makedirs(save_dir, exist_ok=True)
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(save_dir, '%(title)s.%(ext)s'),
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            yt_path = ydl.prepare_filename(info)
            return jsonify({"success": True, "path": yt_path, "title": info['title']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
