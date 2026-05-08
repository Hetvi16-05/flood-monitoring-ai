"""
Crocodile Detector Module
Specialized detector for crocodiles in flood scenarios using pre-trained YOLO model
"""
import torch
from ultralytics import YOLO
import os
from pathlib import Path


class CrocodileDetector:
    """
    Specialized crocodile detector that runs alongside the main YOLO model.
    Uses a pre-trained model trained specifically on crocodile detection.
    """
    
    def __init__(self, model_path=None, device='cpu', conf_threshold=0.3):
        self.device = device
        self.conf_threshold = conf_threshold
        
        print("🌐 Initializing YOLO-World Zero-Shot Crocodile Detector...")
        try:
            # Use the official YOLOv8-World model (High-tech!)
            self.model = YOLO('yolov8s-worldv2.pt') 
            # Expanded vocabulary for better Zero-Shot recall
            self.model.set_classes(["crocodile", "alligator", "large reptile", "caiman"]) 
            self.model.to(self.device)
            print("✅ YOLO-World active: Detecting 'Predators' via Natural Language.")
        except Exception as e:
            print(f"❌ Failed to load YOLO-World: {e}")
            self.model = None
    
    def detect(self, frame, conf_threshold=None):
        """
        Detect crocodiles in frame
        
        Args:
            frame: Input image (numpy array)
            conf_threshold: Optional confidence override
            
        Returns:
            List of detections: [{'box': [x1, y1, x2, y2], 'conf': float, 'type': 'crocodile'}]
        """
        if self.model is None:
            return []
        try:
            # Use override threshold if provided, else fall back to init default
            thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
            results = self.model.predict(frame, conf=thresh, verbose=False)[0]
            detections = []
            
            for box in results.boxes:
                detections.append({
                    'type': 'crocodile',
                    'name': 'crocodile',
                    'box': box.xyxy[0].cpu().numpy().astype(int),
                    'conf': float(box.conf[0])
                })
            
            return detections
        except Exception as e:
            print(f"❌ Crocodile detection error: {e}")
            return []
    
    def has_crocodile(self, frame):
        """
        Quick check if frame contains any crocodile
        
        Args:
            frame: Input image
            
        Returns:
            (has_crocodile: bool, max_conf: float)
        """
        detections = self.detect(frame)
        if not detections:
            return False, 0.0
        
        max_conf = max(d['conf'] for d in detections)
        return True, max_conf


def get_default_crocodile_model_path():
    """Get default path for crocodile detection model"""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "weights" / "crocodile_detector.pt"


if __name__ == "__main__":
    print("🐊 Crocodile Detector Module Ready")
    print("To use: detector = CrocodileDetector(model_path='path/to/model.pt')")
