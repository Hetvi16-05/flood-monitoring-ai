import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEVICE, NUM_CLASSES, MODEL_PATH, YOLO_MODEL_PATH, PROJECT_ROOT
from models.segmentation_v2 import create_deeplabv3plus
from models.crocodile_detector import CrocodileDetector, get_default_crocodile_model_path
from models.depth_estimator import DepthEstimator
from models.temporal_tracker import TemporalFloodTracker

def load_models():
    """
    Production-level model loader.
    Loads:
    1. Custom YOLOv8 (Flood specific)
    2. COCO YOLOv8 (General objects)
    3. DeepLabV3+ (5-channel segmentation)
    4. Crocodile Detector (Specialized)
    5. Depth Estimator (Water depth assessment)
    6. Temporal Flood Tracker (Flood progression analysis)
    """
    # 1. Dual YOLO Initialization
    # We use the same YOLO_MODEL_PATH for coco if not separately defined
    yolo_coco = YOLO(PROJECT_ROOT / "yolov8n.pt")
    
    # Custom YOLO (if path exists, else fallback to coco)
    custom_yolo_path = PROJECT_ROOT / "project" / "weights" / "custom_yolo.pt"
    if os.path.exists(custom_yolo_path):
        yolo_custom = YOLO(custom_yolo_path)
    else:
        yolo_custom = yolo_coco # Fallback for now

    # 2. Advanced Segmentation Model (V2)
    # 5 channels: RGB (3) + Canny (1) + LBP (1)
    seg = create_deeplabv3plus(in_channels=5, num_classes=NUM_CLASSES)
    
    seg_status = "Initialized (Fresh)"
    if os.path.exists(MODEL_PATH):
        try:
            seg.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            seg_status = "Loaded Successfully"
            print(f"✅ Loaded weights from {MODEL_PATH}")
        except Exception as e:
            seg_status = f"Load Error: {e}"
            print(f"⚠️ Warning: Segmentation weights load error: {e}")
    
    seg.to(DEVICE).eval()
    
    # 4. Crocodile Detector (Specialized)
    croc_model_path = get_default_crocodile_model_path()
    croc_detector = CrocodileDetector(model_path=str(croc_model_path), device=str(DEVICE), conf_threshold=0.3)
    
    # 5. Depth Estimator (Water depth assessment)
    depth_estimator = DepthEstimator(model_type='DPT_Hybrid', device=str(DEVICE))
    
    # 6. Temporal Flood Tracker (Flood progression analysis)
    temporal_tracker = TemporalFloodTracker(history_length=30)
    
    # Metadata initialization
    from config import MODEL_VERSION
    metadata = {
        "seg_version": MODEL_VERSION,
        "seg_status": seg_status,
        "device": str(DEVICE),
        "num_classes": NUM_CLASSES,
        "croc_detector": "Loaded" if croc_detector.model else "Not Available",
        "depth_estimator": "Loaded" if depth_estimator.model else "Not Available",
        "temporal_tracker": "Loaded"
    }
    
    return {
        "models": {"yolo_custom": yolo_custom, "yolo_coco": yolo_coco, "seg": seg, "croc_detector": croc_detector, "depth_estimator": depth_estimator, "temporal_tracker": temporal_tracker},
        "metadata": metadata
    }
