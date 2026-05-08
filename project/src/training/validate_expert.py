import torch
import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.model_loader import load_models
from inference.inference_utils import run_hybrid

def run_stress_test():
    print("🔬 RAINWISE PROFESSIONAL VALIDATION ENGINE")
    print("------------------------------------------")
    
    # Force SegFormer mode
    import os
    os.environ["MODEL_TYPE"] = "segformer"
    
    bundle = load_models()
    models = bundle['models']
    
    # Test Scenarios
    test_cases = [
        {"name": "Deep Flood", "path": "project/test_videos/flood_test.mp4", "expected": "HIGH/DANGEROUS"},
        {"name": "Urban Traffic (Dry)", "path": "project/test_videos/pexels_0.mp4", "expected": "LOW"},
        {"name": "Coastal Waves", "path": "project/test_videos/Durban beach closed due to high waves DRAMATIC AERIAL VIDEO.mp4", "expected": "HIGH"},
        {"name": "Typhoon Street", "path": "project/test_videos/Typhoon Gaemi floods streets in Philippines capital ｜ AFP.mp4", "expected": "HIGH/DANGEROUS"}
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n🧪 Testing Scenario: {case['name']}")
        cap = cv2.VideoCapture(case['path'])
        success, frame = cap.read()
        if not success:
            print(f"❌ Failed to load {case['path']}")
            continue
            
        # Run inference
        res_img, water_p, obj_summary, risk_level, risk_score, telemetry = run_hybrid(
            frame, models, show_yolo=True, show_mask=True, explain_ai=True
        )
        
        print(f"✅ Detection Result: {risk_level} ({risk_score}/100)")
        print(f"🌊 Water Coverage: {water_p:.2f}%")
        print(f"🤖 AI Confidence: {telemetry['hybrid_conf']*100:.1f}%")
        
        status = "PASS" if risk_level in case['expected'] else "RE-CHECK"
        results.append({
            "Scenario": case['name'],
            "Risk": risk_level,
            "Water": f"{water_p:.1f}%",
            "Confidence": f"{telemetry['hybrid_conf']*100:.1f}%",
            "Status": status
        })
        cap.release()

    print("\n\n📊 FINAL VALIDATION REPORT")
    print("----------------------------------------------------------------")
    print(f"{'Scenario':<20} | {'Risk':<10} | {'Water':<10} | {'Confidence':<10} | {'Status'}")
    print("-" * 65)
    for r in results:
        print(f"{r['Scenario']:<20} | {r['Risk']:<10} | {r['Water']:<10} | {r['Confidence']:<10} | {r['Status']}")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    run_stress_test()
