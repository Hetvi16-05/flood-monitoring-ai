import os
import cv2
import sys
from pathlib import Path

# Add project/src to sys.path
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "project" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.advanced_inference import AdvancedHybridInference

def test_v3_images():
    print("🚀 Initializing RAINWISE V3 Advanced Engine for testing...")
    engine = AdvancedHybridInference()
    
    img_dir = ROOT / "project" / "dataset_custom" / "images"
    out_dir = ROOT / "project" / "outputs" / "v3_tests"
    os.makedirs(out_dir, exist_ok=True)
    
    images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    if not images:
        print("❌ No images found in dataset_custom/images")
        return
        
    for img_name in images:
        print(f"📷 Testing {img_name}...")
        frame = cv2.imread(str(img_dir / img_name))
        res = engine.process(frame)
        viz = engine.render(frame, res)
        
        cv2.imwrite(str(out_dir / f"v3_viz_{img_name}"), viz)
        print(f"✅ Saved Clean V3 Visualization to {out_dir / f'v3_viz_{img_name}'}")

if __name__ == "__main__":
    test_v3_images()
