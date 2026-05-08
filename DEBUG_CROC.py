import cv2
import torch
from ultralytics import YOLO
import sys
from pathlib import Path

def debug():
    img_path = "/Users/HetviSheth/Flood_Prediction/newmodel/master_crocodile_dataset/train/images/Crocodile-100_jpg.rf.e28391ced58db1d020b2f385d855dcce.jpg"
    print(f"🔍 Debugging Image: {img_path}")
    
    # Load Model
    model = YOLO('yolov8s-worldv2.pt')
    model.set_classes(["crocodile", "alligator", "reptile", "lizard", "animal"])
    
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Error: Could not read image!")
        return

    # Run with NO threshold to see what it thinks
    results = model.predict(img, conf=0.01, verbose=False)[0]
    
    print(f"\n📊 Raw Detections Found: {len(results.boxes)}")
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls_id]
        print(f"   - Identified: {name} | Confidence: {conf:.4f}")

if __name__ == "__main__":
    debug()
