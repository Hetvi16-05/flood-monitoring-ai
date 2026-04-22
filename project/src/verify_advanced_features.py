import cv2
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "project" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from models.model_loader import load_models
    from inference.inference_utils import run_hybrid
    
    print("🚀 Starting Advanced Features Verification...")
    
    # 1. Load models
    model_bundle = load_models()
    models = model_bundle["models"]
    
    # 2. Create a dummy frame (640x480)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 3. Run inference twice (to trigger temporal tracking)
    print("🧪 Running Iteration 1...")
    res_img, water_p, obj_summary, risk_level, risk_score, telemetry1 = run_hybrid(frame, models)
    
    print("🧪 Running Iteration 2...")
    # Change frame slightly for optical flow
    frame2 = frame.copy()
    frame2[100:200, 100:200] = 0
    res_img, water_p, obj_summary, risk_level, risk_score, telemetry2 = run_hybrid(frame2, models)
    
    # 4. Verify telemetry fields
    print("\n📊 Telemetry Verification:")
    features = {
        "Water Depth": telemetry2.get('avg_water_depth'),
        "Expansion Rate": telemetry2.get('expansion_rate'),
        "Predictive Risk": telemetry2.get('predictive_risk_level')
    }
    
    for name, val in features.items():
        if val is not None:
            print(f"✅ {name}: {val}")
        else:
            print(f"❌ {name}: MISSING")
            
    print("\n✅ Verification Complete! No crashes detected.")

except Exception as e:
    print(f"\n❌ Verification Failed with error: {e}")
    import traceback
    traceback.print_exc()
