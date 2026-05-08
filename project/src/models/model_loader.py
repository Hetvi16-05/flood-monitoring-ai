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
from models.predictive_forecaster import PredictiveForecaster
from models.explainability import FloodGradCAM

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
    
    from models.segformer_model import SegFormerFlood
    # Prioritize the Expert weights, then Demo weights
    EXPERT_WEIGHTS = PROJECT_ROOT / "project" / "weights" / "rainwise_v3_1_expert.pth"
    DEMO_WEIGHTS = PROJECT_ROOT / "project" / "weights" / "rainwise_v3_1_demo.pth"
    SEGFORMER_WEIGHTS = EXPERT_WEIGHTS if EXPERT_WEIGHTS.exists() else DEMO_WEIGHTS

    if model_type == "segformer":
        seg = SegFormerFlood(num_classes=2, in_channels=6) # Forced to 2 classes for V3.1
        current_model_path = SEGFORMER_WEIGHTS
        seg_type = "SegFormer-B0 (V3.1 Elite)"
    elif model_type == "swin_transformer" and os.path.exists(TRANSFORMER_MODEL_PATH):
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
    
    # 2b. Explainable AI (Grad-CAM) Initialization
    gcam = None
    try:
        if hasattr(seg, 'backbone'):
            # Target the last feature map of the backbone
            target_layer = seg.backbone.feature_info[-1]['module']
            gcam = FloodGradCAM(seg, target_layer)
            print(f"👁️ XAI Engine: Grad-CAM active for {seg_type}")
    except Exception as e:
        print(f"⚠️ XAI Engine: Could not initialize Grad-CAM: {e}")
    
    # 3. Auxiliary Tools
    croc_detector = CrocodileDetector(model_path=str(get_default_crocodile_model_path()), device=str(DEVICE))
    depth_estimator = DepthEstimator(model_type='DPT_Hybrid', device=str(DEVICE))
    temporal_tracker = TemporalFloodTracker(history_length=30)
    
    # 4. Predictive Forecaster (with LSTM Support)
    forecaster = PredictiveForecaster()
    # Use the upgraded v2 model
    lstm_weights = PROJECT_ROOT / "project" / "weights" / "flood_lstm_v2.pth"
    if not os.path.exists(lstm_weights):
        # Fallback to v1 if v2 isn't ready
        lstm_weights = PROJECT_ROOT / "project" / "weights" / "flood_lstm_v1.pth"
        
    if os.path.exists(lstm_weights):
        forecaster.load_lstm(str(lstm_weights))
    
    from config import MODEL_VERSION
    metadata = {
        "seg_version": seg_type,
        "seg_status": seg_status,
        "device": str(DEVICE),
        "num_classes": NUM_CLASSES,
        "croc_detector": "Loaded",
        "depth_estimator": "Loaded",
        "temporal_tracker": "Loaded",
        "forecaster": "Loaded (LSTM v2)" if forecaster.lstm_model else "Loaded (Rule-based)",
        "xai_engine": "Active" if gcam else "Disabled"
    }
    
    return {
        "models": {
            "yolo_custom": yolo_custom, 
            "yolo_coco": yolo_coco, 
            "seg": seg, 
            "croc_detector": croc_detector, 
            "depth_estimator": depth_estimator, 
            "temporal_tracker": temporal_tracker,
            "forecaster": forecaster,
            "gcam": gcam
        },
        "metadata": metadata
    }
