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
        """
        Initialize crocodile detector
        
        Args:
            model_path: Path to pre-trained crocodile detection model (.pt file)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            conf_threshold: Confidence threshold for crocodile detection
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠️ Crocodile detector model not found. Using fallback detection.")
            self.model = None
    
    def load_model(self, model_path):
        """Load pre-trained crocodile detection model"""
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"✅ Crocodile detector loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load crocodile detector: {e}")
            self.model = None
            return False
    
    def detect(self, frame):
        """
        Detect crocodiles in frame
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            List of detections: [{'box': [x1, y1, x2, y2], 'conf': float, 'type': 'crocodile'}]
        """
        if self.model is None:
            return []
        
        try:
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
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
