import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEVICE, NUM_CLASSES, MODEL_PATH, YOLO_MODEL_PATH, PROJECT_ROOT, TRANSFORMER_MODEL_PATH
from models.segmentation_v2 import create_deeplabv3plus
from models.flood_transformer import SwinFloodNet
from models.crocodile_detector import CrocodileDetector, get_default_crocodile_model_path
from models.depth_estimator import DepthEstimator
from models.temporal_tracker import TemporalFloodTracker

def load_models():
    """
    Production-level model loader.
    """
    # 1. Dual YOLO Initialization
    yolo_coco = YOLO(PROJECT_ROOT / "yolov8n.pt")
    custom_yolo_path = PROJECT_ROOT / "project" / "weights" / "custom_yolo.pt"
    yolo_custom = YOLO(custom_yolo_path) if os.path.exists(custom_yolo_path) else yolo_coco

    # 2. Segmentation Model Selection (Explicit)
    model_type = os.environ.get("MODEL_TYPE", "swin_transformer")
    print(f"📡 Model Selector: Requesting {model_type}...")

    if model_type == "swin_transformer" and os.path.exists(TRANSFORMER_MODEL_PATH):
        seg = SwinFloodNet(num_classes=NUM_CLASSES, in_channels=6)
        current_model_path = TRANSFORMER_MODEL_PATH
        seg_type = "SwinTransformer (V3)"
    else:
        # Load legacy ResNet-based DeepLabV3+
        seg = create_deeplabv3plus(in_channels=5, num_classes=NUM_CLASSES)
        current_model_path = MODEL_PATH
        seg_type = "DeepLabV3+ (V2)"
    
    seg_status = "Initialized (Empty)"
    if os.path.exists(current_model_path):
        try:
            seg.load_state_dict(torch.load(current_model_path, map_location=DEVICE))
            seg_status = f"Loaded Successfully ({seg_type})"
            print(f"✅ Active Model: {seg_type} from {current_model_path.name}")
        except Exception as e:
            seg_status = f"Load Error: {e}"
            print(f"⚠️ Warning: Model load error: {e}")
    
    seg.to(DEVICE).eval()
    
    # 3. Auxiliary Tools
    croc_detector = CrocodileDetector(model_path=str(get_default_crocodile_model_path()), device=str(DEVICE))
    depth_estimator = DepthEstimator(model_type='DPT_Hybrid', device=str(DEVICE))
    temporal_tracker = TemporalFloodTracker(history_length=30)
    
    from config import MODEL_VERSION
    metadata = {
        "seg_version": seg_type,
        "seg_status": seg_status,
        "device": str(DEVICE),
        "num_classes": NUM_CLASSES,
        "croc_detector": "Loaded",
        "depth_estimator": "Loaded",
        "temporal_tracker": "Loaded"
    }
    
    return {
        "models": {
            "yolo_custom": yolo_custom, 
            "yolo_coco": yolo_coco, 
            "seg": seg, 
            "croc_detector": croc_detector, 
            "depth_estimator": depth_estimator, 
            "temporal_tracker": temporal_tracker
        },
        "metadata": metadata
    }
