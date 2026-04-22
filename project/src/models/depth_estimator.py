"""
Water Depth Estimation using MiDaS (Monocular Depth Estimation)
Estimates water depth from single images for enhanced flood risk assessment
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from config import DEVICE


class DepthEstimator:
    """
    Monocular depth estimation using Intel MiDaS model
    Estimates depth from single images for water depth assessment
    """
    
    def __init__(self, model_type='DPT_Hybrid', device=None):
        """
        Initialize depth estimator
        
        Args:
            model_type: 'DPT_Large' (best) or 'DPT_Hybrid' (faster)
            device: Device to run inference on
        """
        self.device = device or DEVICE
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.load_model()
    
    def load_model(self):
        """Load MiDaS depth estimation model"""
        try:
            # Load MiDaS model from torch hub
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, pretrained=True)
            self.model.to(self.device).eval()
            
            # Load appropriate transforms
            if self.model_type == "dpt_large":
                self.transform = torch.hub.load("intel-isl/MiDaS", "transforms")
            else:
                self.transform = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.transform = self.transform.dpt_transform
            
            print(f"✅ MiDaS {self.model_type} loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load MiDaS model: {e}")
            return False
    
    def estimate_depth(self, frame):
        """
        Estimate depth from frame
        
        Args:
            frame: Input image (BGR numpy array)
            
        Returns:
            depth_map: Normalized depth map (0-1)
            depth_values: Depth values in meters (estimated)
        """
        if self.model is None:
            return None, None
        
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_batch = self.transform(img_rgb).to(self.device)
            
            # Predict depth
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy
            depth_map = prediction.cpu().numpy()
            
            # Normalize to 0-1 range
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            
            # Estimate actual depth in meters (approximate scaling)
            # MiDaS outputs relative depth, we estimate absolute depth
            # Assuming max depth in scene is ~10 meters for flood scenarios
            depth_values = depth_map * 10.0
            
            return depth_map, depth_values
            
        except Exception as e:
            print(f"❌ Depth estimation error: {e}")
            return None, None
    
    def get_water_depth(self, frame, water_mask):
        """
        Estimate water depth in flooded areas
        
        Args:
            frame: Input image
            water_mask: Binary mask of water areas (1=water, 0=non-water)
            
        Returns:
            avg_water_depth: Average water depth in meters
            max_water_depth: Maximum water depth in meters
            depth_map: Full depth map
        """
        depth_map, depth_values = self.estimate_depth(frame)
        
        if depth_map is None:
            return 0.0, 0.0, None
        
        # Ensure both water_mask and depth_values have the same shape
        # Resize water_mask to match depth_values
        if water_mask.shape != depth_values.shape:
            water_mask = cv2.resize(water_mask, (depth_values.shape[1], depth_values.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Extract depth values in water regions only
        try:
            water_depths = depth_values[water_mask > 0]
        except IndexError as e:
            print(f"Shape mismatch - water_mask: {water_mask.shape}, depth_values: {depth_values.shape}")
            return 0.0, 0.0, depth_map
        
        if len(water_depths) == 0:
            return 0.0, 0.0, depth_map
        
        avg_water_depth = np.mean(water_depths)
        max_water_depth = np.max(water_depths)
        
        return avg_water_depth, max_water_depth, depth_map
    
    def get_depth_risk_level(self, avg_depth, max_depth):
        """
        Determine risk level based on water depth
        
        Args:
            avg_depth: Average water depth in meters
            max_depth: Maximum water depth in meters
            
        Returns:
            risk_level: 'LOW', 'MEDIUM', 'HIGH', 'DANGEROUS'
            risk_score: Numeric risk score (0-100)
        """
        score = 0
        
        # Average depth contribution
        if avg_depth < 0.3:
            score += 10  # Shallow water
        elif avg_depth < 0.6:
            score += 30  # Knee-deep
        elif avg_depth < 1.2:
            score += 60  # Waist-deep
        else:
            score += 85  # Dangerous depth
        
        # Maximum depth contribution (important for drowning risk)
        if max_depth > 1.5:
            score += 15  # Can submerge a person
        if max_depth > 2.5:
            score += 15  # Can submerge vehicles
        
        score = min(100, score)
        
        if score < 20:
            level = 'LOW'
        elif score < 50:
            level = 'MEDIUM'
        elif score < 80:
            level = 'HIGH'
        else:
            level = 'DANGEROUS'
        
        return level, int(score)


def get_default_depth_model_path():
    """Get default path for depth model (MiDaS is loaded from torch hub)"""
    return None  # MiDaS loaded from torch hub


if __name__ == "__main__":
    print("📏 Water Depth Estimator Module Ready")
    print("To use: estimator = DepthEstimator()")
