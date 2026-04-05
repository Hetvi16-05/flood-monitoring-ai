import os
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn

# -----------------------------
# PATH FIX (Standardized)
# -----------------------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # project/
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (
    MODEL_PATH, 
    YOLO_MODEL_PATH, 
    DEVICE, 
    IMG_SIZE, 
    NUM_CLASSES, 
    CLASSES, 
    COLORS,
    OUTPUT_DIR,
    YOLO_CONF_THRESHOLD,
    WATER_LOW,
    WATER_HIGH,
    WATER_FLOOD,
    ANIMAL_WATER_THRESHOLD
)

# -----------------------------
# HELPERS
# -----------------------------

from utils.risk import get_flood_risk

def load_segmentation_model(path, num_classes, device):
    model = models.segmentation.lraspp_mobilenet_v3_large(num_classes=num_classes)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, 1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, 1)
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def get_risk_level_v2(water_percent, detected_objects):
    """
    Advanced Risk Assessment Logic:
    - LOW: < 10% water, no critical objects
    - MEDIUM: 10-30% water OR 1+ vehicles/animals
    - HIGH: 30-50% water OR person/animal in water
    - DANGEROUS: > 50% water OR person in 20%+ water
    """
    has_person = any(obj['type'] == 'person' for obj in detected_objects)
    has_animal = any(obj['type'] == 'animal' for obj in detected_objects)
    has_vehicle = any(obj['type'] == 'vehicle' for obj in detected_objects)
    
    if water_percent > WATER_FLOOD or (has_person and water_percent > 20):
        return "DANGEROUS", (0, 0, 255)
    elif water_percent > WATER_HIGH or ( (has_person or has_animal) and water_percent > 10):
        return "HIGH", (0, 165, 255)
    elif water_percent > WATER_LOW or has_vehicle or has_animal:
        return "MEDIUM", (0, 255, 255)
    else:
        return "LOW", (0, 255, 0)

def format_objects(detected_objects):
    counts = {}
    for obj in detected_objects:
        t = obj['type']
        counts[t] = counts.get(t, 0) + 1
    
    if not counts: return "None"
    return ", ".join([f"{k}({v})" for k, v in counts.items()])

# -----------------------------
# CORE LOGIC
# -----------------------------

def process_frame(frame, seg_model, yolo_model, mode="IMAGE"):
    h, w = frame.shape[:2]
    
    # 1. YOLO
    yolo_results = yolo_model.predict(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    detected_objects = []
    for box in yolo_results.boxes:
        cls_id = int(box.cls[0])
        name = yolo_model.names[cls_id]
        
        # Categorize
        label = "object"
        if name == 'person': label = "person"
        elif name in ['car', 'motorcycle', 'bus', 'truck']: label = "vehicle"
        elif name in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']: label = "animal"
        elif name in ['house', 'building']: label = "building"
        
        detected_objects.append({
            "type": label,
            "name": name,
            "box": box.xyxy[0].cpu().numpy().astype(int),
            "conf": float(box.conf[0])
        })

    # 2. Segmentation
    img_tensor = cv2.resize(frame, IMG_SIZE)
    img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = seg_model(img_tensor)['out']
        mask = torch.argmax(output[0], dim=0).cpu().numpy().astype(np.uint8)
        
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Flood water is class 0
    water_mask = (mask_resized == 0).astype(np.uint8)
    water_percent = (np.sum(water_mask) / (w * h)) * 100
    
    # 3. Risk & Text
    risk_text = get_flood_risk(
        water_p=water_percent,
        has_person=any(obj['type'] == 'person' for obj in detected_objects),
        has_animal=any(obj['type'] == 'animal' for obj in detected_objects),
        has_vehicle=any(obj['type'] == 'vehicle' for obj in detected_objects),
        total_objects=len(detected_objects)
    )
    
    risk_colors = {
        "LOW": (0, 255, 0),
        "MEDIUM": (0, 255, 255),
        "HIGH": (0, 165, 255),
        "DANGEROUS": (0, 0, 255)
    }
    risk_color = risk_colors.get(risk_text, (255, 255, 255))
    obj_summary = format_objects(detected_objects)
    
    # 4. Rendering
    overlay = frame.copy()
    overlay[water_mask == 1] = [255, 0, 0] # Blue
    output_viz = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    
    # Draw Info Panel (Top-Left)
    cv2.rectangle(output_viz, (0, 0), (450, 150), (0,0,0), -1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output_viz, f"Water: {water_percent:.1f}%", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(output_viz, f"Objects: {obj_summary}", (10, 65), font, 0.6, (255, 255, 255), 2)
    cv2.putText(output_viz, f"Risk: {risk_text}", (10, 105), font, 1.0, risk_color, 3)
    cv2.putText(output_viz, f"Mode: {mode}", (10, 135), font, 0.6, (200, 200, 200), 1)
    
    # Draw Bounding Boxes
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['box']
        cv2.rectangle(output_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_viz, f"{obj['name']}", (x1, y1-5), font, 0.5, (0, 255, 0), 2)
        
    return output_viz

# -----------------------------
# APP MODES
# -----------------------------

def run_live_alert(seg_model, yolo_model, source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Could not open camera source: {source}")
        return

    print(f"🚀 Live Alert Started. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        mode_label = "CCTV LIVE" if source != 0 else "WEBCAM LIVE"
        result = process_frame(frame, seg_model, yolo_model, mode=mode_label)
        
        cv2.imshow("RAINWISE Live Flood Alert", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def run_batch_test(seg_model, yolo_model):
    test_dir = os.path.join(project_root, "project", "test_samples")
    output_dir = os.path.join(OUTPUT_DIR, "hybrid_results")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
        
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🔍 Found {len(images)} test images. Processing...")
    
    for img_name in images:
        frame = cv2.imread(os.path.join(test_dir, img_name))
        result = process_frame(frame, seg_model, yolo_model, mode="BATCH")
        cv2.imwrite(os.path.join(output_dir, img_name), result)
        print(f"✅ Saved results for: {img_name}")

# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    seg_model = load_segmentation_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    yolo_model = YOLO("yolov8n.pt")
    
    print("\n--- RAINWISE HYBRID INFERENCE ---")
    print("1. Run Batch Test (project/test_samples)")
    print("2. Run Live Webcam Mode")
    print("3. Run CCTV Stream Mode (using config info)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        run_batch_test(seg_model, yolo_model)
    elif choice == '2':
        run_live_alert(seg_model, yolo_model, source=0)
    elif choice == '3':
        # Example CCTV feed or use common RTSP placeholder
        url = input("Enter RTSP/HTTP URL (default 0 for webcam): ") or 0
        run_live_alert(seg_model, yolo_model, source=url)
    else:
        print("Invalid choice.")
