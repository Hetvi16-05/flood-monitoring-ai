import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

class FloodClassifierSigLIP:
    """
    RAINWISE Master Gatekeeper.
    Uses the 98.89% Accuracy SigLIP model to validate flood conditions.
    """
    def __init__(self, model_path="project/src/models/flood_classifier_master", device="mps"):
        self.device = device if torch.backends.mps.is_available() else "cpu"
        print(f"🛰️ Initializing Master Gatekeeper on {self.device}...")
        
        # Load from the promoted production folder
        self.processor = AutoImageProcessor.from_pretrained("prithivMLmods/Flood-Image-Detection")
        self.model = AutoModelForImageClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        self.classes = ['flood', 'no_flood']

    def predict(self, frame_bgr):
        """
        Takes a BGR frame from OpenCV and returns (is_flood, confidence)
        """
        # 1. Convert BGR to RGB PIL Image
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 2. Preprocess
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        inputs = {k: v.float() if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # 3. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        # Get scores
        conf, pred_idx = torch.max(probs, dim=1)
        label = self.classes[pred_idx.item()]
        
        # Return True if 'flood' is detected with > 50% confidence
        is_flood = (label == 'flood')
        return is_flood, conf.item()

import cv2 # Required for BGR2RGB
