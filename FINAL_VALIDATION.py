import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path("/Users/HetviSheth/Flood_Prediction")
sys.path.append(str(project_root / "project/src"))

from inference.advanced_inference import AdvancedHybridInference

def validate():
    print("🧪 RAINWISE V3.1 - FINAL SYSTEM VALIDATION")
    print("------------------------------------------")
    
    # Initialize Engine
    engine = AdvancedHybridInference()
    
    tests = [
        {
            "name": "🐊 PREDATOR DETECTION",
            "path": "/Users/HetviSheth/Flood_Prediction/newmodel/master_crocodile_dataset/train/images/Crocodile-100_jpg.rf.e28391ced58db1d020b2f385d855dcce.jpg",
            "expected": "EXTREME"
        },
        {
            "name": "🌵 DRY LAND VALIDATION",
            "path": "/Users/HetviSheth/Flood_Prediction/newmodel/classification_dataset/valid/no_flood/03e1159a87b345a087c0f2bc548f53e5.jpg",
            "expected": "LOW"
        },
        {
            "name": "🌊 CATASTROPHIC FLOOD",
            "path": "/Users/HetviSheth/Flood_Prediction/newmodel/classification_dataset/valid/flood/25441115_jpg.rf.c9f03b40de30dd06333ddfbb8b00d00a.jpg",
            "expected": "HIGH/EXTREME"
        }
    ]
    
    for test in tests:
        print(f"\n▶ Running: {test['name']}")
        if not os.path.exists(test['path']):
            print(f"❌ Missing file: {test['path']}")
            continue
            
        img = cv2.imread(test['path'])
        res = engine.process(img)
        
        print(f"   Result Risk: {res['risk_level']} ({res['risk_score']:.1f}/100)")
        print(f"   Water Area: {res['water_p']:.1f}%")
        print(f"   Croc Detected: {res.get('croc_detected', False)}")
        
        # Simple Logic Check
        if test['expected'] in res['risk_level'] or (test['expected'] == "HIGH/EXTREME" and res['risk_score'] > 70):
            print(f"   ✅ PASS: Matches expected profile.")
        else:
            print(f"   ⚠️ WARNING: Expected {test['expected']} but got {res['risk_level']}.")

if __name__ == "__main__":
    validate()
