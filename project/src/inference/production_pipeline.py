import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class RainwiseProductionPipeline:
    """
    Production-grade inference pipeline for RAINWISE V3.
    Features: Hybrid Water Detection, Temporal Tracking, Infrastructure Impact.
    """
    def __init__(self, model_path, device='mps'):
        self.device = device
        # Load YOLOv8 for base segmentation/detection
        self.model = YOLO(model_path)
        self.prev_water_area = 0
        self.growth_history = []
        
        # Risk thresholds
        self.RISK_LEVELS = {
            "NONE": (0, 5, (0, 255, 0)),
            "LOW": (5, 15, (0, 255, 255)),
            "MEDIUM": (15, 30, (0, 165, 255)),
            "HIGH": (30, 50, (0, 0, 255)),
            "EXTREME": (50, 100, (0, 0, 139))
        }

    def compute_hybrid_mask(self, frame):
        """
        Combines AI, Spectral (NDWI), and Texture signals.
        """
        # 1. AI Prediction
        results = self.model(frame, device=self.device, verbose=False)
        ai_mask = np.zeros(frame.shape[:2], dtype=np.float32)
        
        # Assuming class '0' is water in the trained YOLO model
        for result in results:
            if result.masks is not None:
                for mask, cls in zip(result.masks.data, result.boxes.cls):
                    if cls == 0: # water
                        ai_mask += mask.cpu().numpy()

        # 2. Spectral Wetness ( (B-R)/(B+R) )
        b, g, r = cv2.split(frame.astype(np.float32))
        wetness = (b - r) / (b + r + 1e-6)
        wetness = cv2.threshold(wetness, 0.1, 1, cv2.THRESH_BINARY)[1]

        # 3. Texture Rejection (Sobel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        texture = np.abs(sobel)
        low_texture = cv2.threshold(texture, 50, 1, cv2.THRESH_BINARY_INV)[1]

        # 4. Fusion
        final_mask = (ai_mask * 0.6) + (wetness * 0.3) + (low_texture * 0.1)
        return cv2.threshold(final_mask, 0.5, 1, cv2.THRESH_BINARY)[1]

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Core Detection
            water_mask = self.compute_hybrid_mask(frame)
            water_area = np.sum(water_mask) / (w * h) * 100 # % of frame
            
            # Temporal Tracking
            growth = water_area - self.prev_water_area if frame_idx > 0 else 0
            self.prev_water_area = water_area
            self.growth_history.append(growth)
            
            # Risk Assessment
            risk = "NONE"
            color = (0, 255, 0)
            for level, (low, high, c) in self.RISK_LEVELS.items():
                if low <= water_area < high:
                    risk = level
                    color = c
                    break
            
            # Visualization Overlay
            # 1. Semi-transparent water mask (Blue)
            overlay = frame.copy()
            overlay[water_mask > 0] = [255, 128, 0] # BGR Blue
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # 2. Telemetry HUD
            cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)
            cv2.putText(frame, f"RAINWISE V3 INTELLIGENCE", (20, 40), 2, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"RISK LEVEL: {risk}", (20, 80), 2, 0.8, color, 2)
            cv2.putText(frame, f"WATER AREA: {water_area:.1f}%", (20, 110), 2, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"GROWTH RATE: {growth:+.2f}%/f", (20, 140), 2, 0.6, (255, 255, 255), 1)
            
            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        print(f"🎬 Processed video saved to: {output_path}")

if __name__ == "__main__":
    # Example usage (requires a trained YOLO model)
    # pipeline = RainwiseProductionPipeline(model_path="weights/yolov8n-seg.pt")
    # pipeline.process_video("test_video.mp4", "output_analysis.mp4")
    print("🚀 Rainwise Production Pipeline Ready.")
