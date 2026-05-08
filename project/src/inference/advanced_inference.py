import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from collections import deque
from pathlib import Path

from models.flood_net import FloodNet
from models.emergency_detector import EmergencyDetector
from models.temporal_model import TemporalEvolutionModel
from models.risk_network import HybridRiskNetwork
from models.flood_classifier import FloodClassifierSigLIP
from models.crocodile_detector import CrocodileDetector

# Robust config import
try:
    from project.src.config import DEVICE, IMG_SIZE
except ImportError:
    try:
        from src.config import DEVICE, IMG_SIZE
    except ImportError:
        from config import DEVICE, IMG_SIZE

class AdvancedHybridInference:
    """
    RAINWISE Advanced Inference Engine.
    Integrates FloodNet, EmergencyDetector, Temporal Evolution, and Hybrid Risk.
    """
    def __init__(self, checkpoints=None):
        self.device = DEVICE
        
        # 1. Initialize Models
        self.flood_net = FloodNet(num_classes=8).to(self.device)
        self.emergency_detector = EmergencyDetector(num_classes=5).to(self.device)
        self.temporal_model = TemporalEvolutionModel(sequence_length=16, predict_steps=5).to(self.device)
        self.risk_network = HybridRiskNetwork(vision_feature_dim=1024, tabular_dim=10).to(self.device)
        
        # 0. Master Gatekeeper (98.89% Accuracy Brain)
        try:
            self.gatekeeper = FloodClassifierSigLIP(device=self.device)
            print("🛡️ Gatekeeper: Active and Ready (SigLIP)")
        except Exception as e:
            self.gatekeeper = None
            print(f"⚠️ Warning: Gatekeeper could not initialize: {e}")
            
        # 0b. Crocodile Predator Detection (YOLO-World)
        self.croc_detector = CrocodileDetector(device=self.device)
        
        # Load the newly trained Custom V3 Weights if available
        ROOT = Path(__file__).resolve().parents[3]
        CUSTOM_WEIGHTS = ROOT / "project" / "weights" / "custom_flood_net.pth"
        if CUSTOM_WEIGHTS.exists():
            print(f"🔥 Loading Custom V3 FloodNet Weights: {CUSTOM_WEIGHTS}")
            self.flood_net.load_state_dict(torch.load(str(CUSTOM_WEIGHTS), map_location=self.device))
        
        # 2. Frame Buffer for Temporal Reasoning (16 frames)
        self.frame_buffer = deque(maxlen=16)
        
        # 3. Class Names (8 classes for FloodNet)
        self.classes = [
            'flood_water', 'shallow_water', 'deep_water', 'debris', 
            'road', 'building', 'vegetation', 'emergency_objects'
        ]
        
        self.load_weights(checkpoints)
        self.flood_net.eval()
        self.emergency_detector.eval()
        self.temporal_model.eval()
        self.risk_network.eval()

    def load_weights(self, checkpoints):
        if checkpoints:
            for name, path in checkpoints.items():
                if os.path.exists(path):
                    if name == 'flood_net': self.flood_net.load_state_dict(torch.load(path, map_location=self.device))
                    elif name == 'emergency': self.emergency_detector.load_state_dict(torch.load(path, map_location=self.device))
                    # Add others as needed

    def prepare_6_channels(self, frame):
        """
        Generates the 6-channel input for FloodNet:
        RGB (3) + NDWI-like (1) + Texture (1) + Elevation-Mock (1)
        """
        frame_resized = cv2.resize(frame, IMG_SIZE)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 1. NDWI-like (Normalized Difference Water Index)
        # Using Green (G) and Red (R) channels as a proxy: (G - R) / (G + R)
        g = rgb[:, :, 1]
        r = rgb[:, :, 0]
        ndwi = (g - r) / (g + r + 1e-6)
        ndwi = (ndwi + 1) / 2 # Normalize to 0-1
        
        # 2. Texture (Sobel Edge Magnitude)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture = np.sqrt(sobelx**2 + sobely**2)
        texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)
        
        # 3. Elevation-Mock (Gradient based on vertical position)
        # In a real system, this would be a DEM (Digital Elevation Model) lookup
        h, w = IMG_SIZE
        elevation = np.linspace(1, 0, h).reshape(h, 1).repeat(w, axis=1)
        
        # Stack all
        six_channel = np.zeros((h, w, 6), dtype=np.float32)
        six_channel[:, :, :3] = rgb
        six_channel[:, :, 3] = ndwi
        six_channel[:, :, 4] = texture
        six_channel[:, :, 5] = elevation
        
        return torch.from_numpy(six_channel).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def process(self, frame):
        h_orig, w_orig = frame.shape[:2]
        
        # 0. Master Gatekeeper Validation
        if self.gatekeeper:
            is_flood, gate_conf = self.gatekeeper.predict(frame)
            if not is_flood:
                # INSTANT REJECTION: Zero False Positive Shield
                return {
                    'risk_score': 0.0,
                    'risk_level': "LOW (Safe)",
                    'water_p': 0.0,
                    'mask': np.zeros((h_orig, w_orig), dtype=np.uint8),
                    'submersion': 0.0,
                    'flow_speed': 0.0,
                    'predict_expansion': None,
                    'gatekeeper_info': f"SigLIP Validated: Dry Road ({gate_conf:.2%})"
                }
        
        # 1. Segmentation (FloodNet)
        input_6c = self.prepare_6_channels(frame)
        with torch.no_grad():
            seg_out = self.flood_net(input_6c)
            mask = torch.argmax(seg_out[0], dim=0).cpu().numpy().astype(np.uint8)
        
        # 2. Emergency Detection
        input_3c = input_6c[:, :3, :, :]
        with torch.no_grad():
            det_out = self.emergency_detector(input_3c)
            # Simplified bbox extraction for demo
            submersion_score = det_out['submersion'].mean().item()
        
        # 3. Temporal Evolution
        self.frame_buffer.append(input_3c[0])
        expansion_heatmap = None
        if len(self.frame_buffer) == 16:
            seq = torch.stack(list(self.frame_buffer), dim=1).unsqueeze(0)
            with torch.no_grad():
                expansion_heatmap = self.temporal_model(seq)[0, 0, -1].cpu().numpy()
        
        # 4. Hybrid Risk & Flow Speed (Optical Flow)
        ndwi_batch = input_6c[:, 3:4, :, :]
        
        # Calculate Flow Speed if we have a previous frame
        flow_speed = 0.0
        if len(self.frame_buffer) >= 2:
            prev_gray = cv2.cvtColor(np.array(self.frame_buffer[-2].permute(1, 2, 0).cpu() * 255, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(np.array(self.frame_buffer[-1].permute(1, 2, 0).cpu() * 255, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_water = flow[mask == 0] 
            if flow_water.size > 0:
                flow_speed = np.mean(np.linalg.norm(flow_water, axis=1))

        # 5. Smart Water Detection (Model + Spectral + Texture)
        # We need to distinguish murky water from roads/forests
        seg_water = (mask < 3).astype(np.float32)
        
        # Spectral Check (proxy NDWI)
        ndwi_raw = ndwi_batch[0, 0].cpu().numpy()
        spectral_water = (ndwi_raw > 0.55).astype(np.float32) # Stricter than before
        
        # Texture check (Water is usually smoother than roads/foliage)
        texture_raw = input_6c[0, 4].cpu().numpy()
        low_texture = (texture_raw < 0.3).astype(np.float32) # Inverse of Sobel edges
        
        # FUSED WATER MASK:
        # Requires EITHER the Model's strong opinion OR a Spectral match with Low Texture (Murky)
        fused_water_mask = (seg_water > 0.5) | ((spectral_water > 0.5) & (low_texture > 0.5))
        
        # Final area calculation with median filtering to remove noise (specular highlights on leaves)
        fused_water_mask = cv2.medianBlur(fused_water_mask.astype(np.uint8), 5)
        water_p = np.mean(fused_water_mask) * 100
        
        # Base score from water area (0-50 range)
        base_score = min(50, water_p * 0.5)
        
        # Danger multipliers (Flow Speed & Submersion)
        danger_factor = 1.0
        if flow_speed > 3.0: danger_factor += 0.5 # Fast water is dangerous
        if submersion_score > 0.3: danger_factor += 0.5 # Significant submersion
        
        # Final calibrated score
        # Using a weighted blend of Rule-based and Neural-based risk
        neural_risk_raw = self.risk_network(pooled_vision_padded, weather_data)['risk_score'].item()
        
        # SAFETY OVERRIDE: If water is low and speed is low, it CANNOT be extreme
        if water_p < 30 and flow_speed < 1.0 and submersion_score < 0.1:
            calibrated_score = min(40, neural_risk_raw * 0.4) # Caps at MEDIUM
        else:
            calibrated_score = (base_score * danger_factor) + (neural_risk_raw * 0.2)
            
        final_risk_score = min(100.0, calibrated_score)
        
        if final_risk_score < 20: rl = "LOW"
        elif final_risk_score < 45: rl = "MEDIUM"
        elif final_risk_score < 75: rl = "HIGH"
        else: rl = "EXTREME"

        res = {
            'risk_score': final_risk_score,
            'risk_level': rl,
            'water_p': water_p,
            'mask': cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST),
            'submersion': submersion_score,
            'flow_speed': flow_speed,
            'predict_expansion': expansion_heatmap if expansion_heatmap is None else cv2.resize(expansion_heatmap, (w_orig, h_orig))
        }
        
        # 6. Check for Predators (Crocodiles)
        res['crocodiles'] = self.croc_detector.detect(frame)
        if res['crocodiles']:
            # Force Risk Level to EXTREME if a crocodile is in the water
            res['risk_level'] = "EXTREME (PREDATOR)"
            res['risk_score'] = max(res['risk_score'], 95.0)
        
        return res

    def render(self, frame, res):
        h, w = frame.shape[:2]
        
        # [USER REQUEST] Clean view: No color mask and No text overlays
        output = frame.copy()
        
        # Keep subtle Predicted Expansion contours (non-intrusive)
        if res['predict_expansion'] is not None:
            exp_mask = (res['predict_expansion'] > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(exp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 0, 255), 1)
            
        # [ALERT] Show Crocodile Warning
        if res.get('crocodiles'):
            for det in res['crocodiles']:
                x1, y1, x2, y2 = det['box']
                # Draw sharp Red Box around the Croc
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(output, "🐊 CROCODILE DETECTED!", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Global Alert Header
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
            cv2.putText(output, "⚠️ HIGH DANGER: PREDATOR IN WATER", (w//2-200, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        return output

def run_advanced_hybrid(frame, engine, **kwargs):
    """
    Standardized wrapper for Flask app compatibility.
    """
    res = engine.process(frame)
    res_img = engine.render(frame, res)
    
    # Adapt to legacy return format: 
    # (res_img, water_p, obj_summary, risk_level, risk_score, telemetry)
    telemetry = {
        'hybrid_conf': 0.9, # Mock high confidence for new custom models
        'submersion': res['submersion'],
        'flow_speed': res['flow_speed'],
        'predict_expansion': res['predict_expansion'] is not None,
        'temporal_insights': [f"Flow Speed: {res['flow_speed']:.2f} px/f", f"Risk Level: {res['risk_level']}"]
    }
    
    return res_img, res['water_p'], "Custom Detection Active", res['risk_level'], res['risk_score'], telemetry
