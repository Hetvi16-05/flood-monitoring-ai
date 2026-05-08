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
models_bundle = None
last_telemetry = {}
risk_history = []
conf_history = []
user_settings = {
    "show_yolo": True,
    "show_mask": True,
    "enable_sound": True,
    "enable_email": False,
    "explain_ai": False,
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

def get_models():
    global models_bundle
    if models_bundle is None:
        print("🛠️ Initializing RAINWISE Model Bundle...")
        models_bundle = load_models()
    return models_bundle

@app.route('/')
def index():
    try:
        lat, lon, city = get_location()
        rain = get_rainfall(lat, lon)
    except:
        lat, lon, city, rain = 0, 0, "Unknown", 0
    
    source = request.args.get('source', '0')
    return render_template('index.html', city=city, rain=rain, lat=lat, lon=lon, current_source=source)

@app.route('/video_feed')
def video_feed():
    source = request.args.get('source', '0')
    # Convert to int if it's a webcam index
    if source.isdigit():
        source = int(source)
    return Response(gen_frames(camera_source=source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test_image', methods=['POST'])
def test_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    bundle = get_models()
    res_img, water_p, _, risk_level, risk_score, telemetry = run_advanced_hybrid(frame, bundle['engine'])
    
    # Return results for display
    return jsonify({
        "risk_level": risk_level,
        "risk_score": risk_score,
        "water_p": water_p,
        "telemetry": telemetry
    })

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
    bundle = get_models()
    models = bundle['models']
    
    frame_count = 0
    # Higher frame skip = Cooler Mac
    FRAME_SKIP = 5 
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if frame_count % FRAME_SKIP == 0:
            # [UPGRADED] Using Advanced V3.1 Elite Engine
            try:
                res_img, water_p, obj_summary, risk_level, risk_score, telemetry = run_advanced_hybrid(
                    frame, bundle['engine']
                )
                
                # [NEW] MASTER ALERT LOGGING
                if telemetry.get('croc_detected'):
                    print(f"🚨 🐊 MASTER ALERT: {telemetry['croc_count']} CROCODILE(S) DETECTED!")
                
                # IMPORTANT: Update global state
                telemetry['risk_level'] = risk_level
                telemetry['risk_score'] = risk_score
                telemetry['water_p'] = water_p
                
                # Use forecaster for temporal prediction
                forecaster = models['forecaster']
                lat, lon, _ = get_location()
                rain = get_rainfall(lat, lon)
                
                prediction = forecaster.predict_flood_risk(
                    lat=lat, lon=lon, 
                    current_risk_score=risk_score,
                    current_water_p=water_p,
                    current_rain=rain
                )
                
                telemetry['prediction'] = prediction
                telemetry['early_warning'] = forecaster.get_early_warning(prediction)
                
                # Critical Emergency Alert
                if prediction.get('emergency'):
                    telemetry['emergency_alert'] = prediction['emergency']
                
                last_telemetry = telemetry
                
                risk_history.append(risk_score)
                if len(risk_history) > 50: risk_history.pop(0)
                
                conf_history.append(telemetry['hybrid_conf'])
                if len(conf_history) > 50: conf_history.pop(0)

                # Log to CSV
            except Exception as e:
                print(f"❌ Inference Error: {e}")
                import traceback
                traceback.print_exc()
                res_img = frame # Fallback to raw frame
        else:
            # Just use the raw frame or previous results
            res_img = frame

        frame_count += 1
        
        # Encode and yield
        ret, buffer = cv2.imencode('.jpg', res_img)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Artificial delay to cool down CPU
        time.sleep(0.01)

@app.route('/update_settings', methods=['POST'])

@app.route('/predict', methods=['POST'])
def predict():
    global last_telemetry, risk_history, conf_history
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        bundle = get_models()
        models = bundle['models']
        
        res_img, water_p, obj_summary, risk_level, risk_score, telemetry = run_hybrid(
            frame, models,
            show_yolo=user_settings['show_yolo'],
            show_mask=user_settings['show_mask'],
            explain_ai=user_settings['explain_ai']
        )
        
        # Update state
        last_telemetry = telemetry
        
        # Use forecaster for prediction
        forecaster = models['forecaster']
        lat, lon, _ = get_location()
        rain = get_rainfall(lat, lon)
        prediction = forecaster.predict_flood_risk(
            lat=lat, lon=lon, 
            current_risk_score=risk_score,
            current_water_p=water_p,
            current_rain=rain
        )
        
        last_telemetry['risk_level'] = str(risk_level)
        last_telemetry['risk_score'] = float(risk_score)
        last_telemetry['water_p'] = float(water_p)
        last_telemetry['prediction'] = prediction
        last_telemetry['early_warning'] = forecaster.get_early_warning(prediction)
        if prediction.get('emergency'):
            last_telemetry['emergency_alert'] = prediction['emergency']
        
        # Log to CSV
        try:
            lat, lon, city = get_location()
            rain = get_rainfall(lat, lon)
            log_to_csv(water_p, obj_summary, risk_level, risk_score, rain, city)
        except:
            pass
        
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

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        import requests
        from io import BytesIO
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        bundle = get_models()
        res_img, water_p, _, risk_level, risk_score, telemetry = run_advanced_hybrid(frame, bundle['engine'])
        
        _, buffer = cv2.imencode('.jpg', res_img)
        import base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify(sanitize_data({
            "result_image": img_base64,
            "telemetry": telemetry,
            "risk_level": str(risk_level),
            "risk_score": float(risk_score),
            "water_p": float(water_p)
        }))
    except Exception as e:
        return jsonify({"error": f"URL analysis failed: {str(e)}"}), 500

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
