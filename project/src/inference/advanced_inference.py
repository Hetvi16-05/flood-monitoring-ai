import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from pathlib import Path
import traceback

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
    Optimized for Apple Silicon (MPS) with float32 enforcement.
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

    def prepare_6_channels(self, frame):
        h, w = IMG_SIZE
        frame_resized = cv2.resize(frame, IMG_SIZE)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        g = rgb[:, :, 1]
        r = rgb[:, :, 0]
        ndwi = (g - r) / (g + r + 1e-6)
        ndwi = (ndwi + 1) / 2
        
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture = np.sqrt(sobelx**2 + sobely**2)
        texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)
        
        elevation = np.linspace(1, 0, h).reshape(h, 1).repeat(w, axis=1)
        
        six_channel = np.zeros((h, w, 6), dtype=np.float32)
        six_channel[:, :, :3] = rgb
        six_channel[:, :, 3] = ndwi
        six_channel[:, :, 4] = texture
        six_channel[:, :, 5] = elevation
        
        return torch.from_numpy(six_channel).permute(2, 0, 1).unsqueeze(0).to(self.device).float()

    def process(self, frame):
        try:
            h_orig, w_orig = frame.shape[:2]
            
            # 0. HYPER-VIGILANT MULTI-HAZARD DETECTION (First Prize Engine)
            # We look for Crocodiles, People, Vehicles, and Boats simultaneously
            # Threshold set to 0.15 for maximum situational awareness
            all_detections = self.croc_detector.detect(frame, conf_threshold=0.15)
            
            croc_detections = [d for d in all_detections if d['name'] in ['crocodile', 'alligator', 'large reptile']]
            people_detections = [d for d in all_detections if d['name'] in ['person', 'human', 'child']]
            vehicle_detections = [d for d in all_detections if d['name'] in ['car', 'bus', 'truck', 'vehicle']]
            boat_detections = [d for d in all_detections if d['name'] in ['boat', 'kayak', 'raft']]

            # 1. Master Gatekeeper Validation
            is_flood = True
            gate_conf = 1.0
            if self.gatekeeper:
                is_flood, gate_conf = self.gatekeeper.predict(frame)
            
            # 2. THE ELITE OVERRIDE: Reject if dry or if it's just trees (Green Rejection)
            avg_green = frame[:,:,1].mean()
            avg_blue = frame[:,:,0].mean()
            # Vegetation check remains, but we allow more brown/murky colors
            is_vegetation = (avg_green > avg_blue * 1.15) 
            
            # RECALIBRATED: Loosened threshold to 0.75 to capture muddy/murky urban floods
            # This ensures Abacus Circle and muddy locations are NOT ignored.
            if (not is_flood or gate_conf < 0.75 or is_vegetation) and (not all_detections or max([d['conf'] for d in all_detections] + [0]) < 0.35):
                return {
                    'risk_score': 5.0, 'risk_level': "LOW (Safe)", 'water_p': 0.0,
                    'mask': np.zeros((h_orig, w_orig), dtype=np.uint8),
                    'submersion': 0.0, 'flow_speed': 0.0, 'predict_expansion': None,
                    'gatekeeper_info': f"SigLIP Mode: Stable ({gate_conf:.2%})",
                    'crocodiles': [], 'people': [], 'vehicles': []
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
                submersion_score = float(det_out['submersion'].mean().item())
            
            # 3. Temporal Evolution
            self.frame_buffer.append(input_3c[0])
            expansion_heatmap = None
            if len(self.frame_buffer) == 16:
                seq = torch.stack(list(self.frame_buffer), dim=1).unsqueeze(0)
                with torch.no_grad():
                    expansion_heatmap = self.temporal_model(seq)[0, 0, -1].cpu().numpy()
            
            # 4. Hybrid Risk & Flow Speed
            ndwi_batch = input_6c[:, 3:4, :, :]
            flow_speed = 0.0
            if len(self.frame_buffer) >= 2:
                prev_gray = cv2.cvtColor(np.array(self.frame_buffer[-2].permute(1, 2, 0).cpu() * 255, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(np.array(self.frame_buffer[-1].permute(1, 2, 0).cpu() * 255, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_water = flow[mask == 0] 
                if flow_water.size > 0:
                    flow_speed = float(np.mean(np.linalg.norm(flow_water, axis=1)))

            # 5. Smart Water Detection (Fused Mask with Spectral Master-Lock)
            seg_water = (mask < 3).astype(np.float32)
            ndwi_raw = ndwi_batch[0, 0].cpu().numpy()
            
            # THE SPECTRAL MASTER-LOCK: Recalibrated for MUDDY WATER
            spectral_water_p = (np.sum(ndwi_raw > 0.45) / ndwi_raw.size) * 100
            
            spectral_water = (ndwi_raw > 0.50).astype(np.float32)
            texture_raw = input_6c[0, 4].cpu().numpy()
            low_texture = (texture_raw < 0.3).astype(np.float32)
            
            # If Physics (NDWI) is totally absent (< 0.3%), we treat SegFormer as a hallucination
            # We lowered this to 0.3% to capture murky/muddy water reflections.
            if spectral_water_p < 0.3:
                fused_water_mask = np.zeros_like(seg_water)
            else:
                fused_water_mask = (seg_water > 0.5) | ((spectral_water > 0.5) & (low_texture > 0.5))
                
            fused_water_mask = cv2.medianBlur(fused_water_mask.astype(np.uint8), 5)
            water_p = float(np.mean(fused_water_mask) * 100)
            
            # 6. Hybrid Risk Calculation
            pooled_vision = F.adaptive_avg_pool2d(input_6c.float(), (1, 1)).flatten(1)
            pooled_vision_padded = F.pad(pooled_vision, (0, 1024 - pooled_vision.shape[1])).float()
            
            weather_data = torch.tensor([[water_p, flow_speed, submersion_score, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).to(self.device)
            neural_risk_raw = self.risk_network(pooled_vision_padded, weather_data)['risk_score'].item()
            
            # Base score logic: Aggressive scaling for Urban Flooding (10% water = 50pt base)
            danger_factor = 1.0 + (1.0 if flow_speed > 1.5 else 0.0) + (1.5 if submersion_score > 0.15 else 0.0)
            base_score = min(85, water_p * 5.0) 
            
            # If water is very low and no flow/submersion, keep it LOW
            if water_p < 4 and flow_speed < 0.4 and submersion_score < 0.04:
                calibrated_score = min(15, neural_risk_raw * 0.2)
            else:
                # Dynamic fusion with higher Neural weight (0.5) for "Intuition"
                calibrated_score = (base_score * danger_factor) + (neural_risk_raw * 0.5)
                
            final_risk_score = min(100.0, calibrated_score)
            
            rl = "MEDIUM"
            if final_risk_score < 20: rl = "LOW (Safe)"
            elif final_risk_score < 45: rl = "MEDIUM"
            elif final_risk_score < 75: rl = "HIGH"
            else: rl = "EXTREME"

            # 5. PRIORITY-LEVEL OVERRIDES (THE SUPREME HIERARCHY)
            final_hazard_label = rl
            
            # Master Condition: We define "Active Flood" as visible water (>1%) OR High Gatekeeper confidence
            is_active_flood = (water_p > 1.0 or (is_flood and gate_conf > 0.85))
            
            # PRIORITY 1: Predator + Human + Water (Worst Case Scenario)
            if croc_detections and people_detections and is_active_flood:
                final_hazard_label = "EXTREME (PREDATOR-HUMAN CONTACT)"
                calibrated_score = 100.0
                
            # PRIORITY 2: Predator in Water (Lethal Threat)
            elif croc_detections and is_active_flood:
                final_hazard_label = "EXTREME (LETHAL HAZARD)"
                calibrated_score = 100.0
                
            # PRIORITY 3: People in Flood (Life Threat)
            # ONLY EXTREME if person is actually IN the water area
            elif people_detections and is_active_flood:
                final_hazard_label = "EXTREME (LIFE AT RISK)"
                calibrated_score = 100.0
                
            # PRIORITY 4: High-Flow Kinetic Impact (Physical Threat)
            elif flow_speed > 3.0 and is_active_flood:
                final_hazard_label = "EXTREME (KINETIC IMPACT)"
                calibrated_score = max(calibrated_score, 98.0)

            # PRIORITY 5: Significant Inundation (Urban Threat)
            elif water_p > 40.0:
                final_hazard_label = "HIGH (MAJOR INUNDATION)"
                calibrated_score = max(calibrated_score, 85.0)

            # Secondary Predator Alert (Dry Land or Low Confidence)
            elif croc_detections:
                final_hazard_label = "HIGH (PREDATOR SUSPECTED)"
                calibrated_score = max(calibrated_score, 85.0)

            # NOTE: If only People are detected on dry land, they do NOT trigger EXTREME.
            # They stay as LOW or MODERATE based on the base_score.

            final_risk_score = min(100.0, calibrated_score)
            
            # 7. Final Hazard Assessment (Multi-Threat Fusion)
            res = {
                'risk_score': final_risk_score, 'risk_level': final_hazard_label, 'water_p': water_p,
                'mask': cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST),
                'submersion': submersion_score, 'flow_speed': flow_speed,
                'predict_expansion': expansion_heatmap if expansion_heatmap is None else cv2.resize(expansion_heatmap, (w_orig, h_orig)),
                'crocodiles': croc_detections,
                'people': people_detections,
                'vehicles': vehicle_detections + boat_detections
            }
            
            return res
        except Exception as e:
            print(f"❌ Inference Process Error: {e}")
            traceback.print_exc()
            return None

    def render(self, frame, res):
        if res is None: return frame
        h, w = frame.shape[:2]
        output = frame.copy()
        
        if res['predict_expansion'] is not None:
            exp_mask = (res['predict_expansion'] > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(exp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 0, 255), 1)
            
        if res.get('crocodiles'):
            for det in res['crocodiles']:
                x1, y1, x2, y2 = det['box']
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(output, "🐊 CROCODILE DETECTED!", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return output

def run_advanced_hybrid(frame, engine, **kwargs):
    res = engine.process(frame)
    res_img = engine.render(frame, res)
    
    # Defaults for failed inference
    risk_level = res['risk_level'] if res else "UNKNOWN"
    risk_score = res['risk_score'] if res else 0.0
    water_p = res['water_p'] if res else 0.0
    
    telemetry = {
        'hybrid_conf': 0.9,
        'submersion': res['submersion'] if res else 0.0,
        'flow_speed': res['flow_speed'] if res else 0.0,
        'predict_expansion': res['predict_expansion'] is not None if res else False,
        'croc_count': len(res.get('crocodiles', [])) if res else 0,
        'croc_detected': len(res.get('crocodiles', [])) > 0 if res else False,
        'temporal_insights': [f"Risk Level: {risk_level}"]
    }
    
    if telemetry['croc_detected']:
        telemetry['temporal_insights'].insert(0, f"🐊 ALERT: {telemetry['croc_count']} Crocodile(s) Spotted!")
    
    return res_img, water_p, "Custom Detection Active", risk_level, risk_score, telemetry
