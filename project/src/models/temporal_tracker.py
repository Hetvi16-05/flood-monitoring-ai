"""
Temporal Flood Tracking using Optical Flow
Tracks flood progression over time using optical flow and temporal analysis
"""
import cv2
import numpy as np
from collections import deque
from config import INF_SIZE


class TemporalFloodTracker:
    """
    Tracks flood progression over time using optical flow
    Analyzes flood spread direction, speed, and predicts future expansion
    """
    
    def __init__(self, history_length=30):
        """
        Initialize temporal flood tracker
        
        Args:
            history_length: Number of frames to keep in history for analysis
        """
        self.history_length = history_length
        self.water_mask_history = deque(maxlen=history_length)
        self.flow_history = deque(maxlen=history_length)
        self.prev_frame = None
        
        # Optical flow parameters
        self.farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def update(self, frame, water_mask):
        """
        Update tracker with new frame and water mask
        
        Args:
            frame: Current frame (BGR)
            water_mask: Binary water mask (1=water, 0=non-water)
            
        Returns:
            flow_data: Optical flow data
            progression_stats: Flood progression statistics
        """
        # Store water mask
        self.water_mask_history.append(water_mask.copy())
        
        # Calculate optical flow
        flow_data = self._calculate_optical_flow(frame)
        
        # Analyze flood progression
        progression_stats = self._analyze_progression()
        
        self.prev_frame = frame.copy()
        return flow_data, progression_stats
    
    def _calculate_optical_flow(self, frame):
        """
        Calculate optical flow using Farneback method
        
        Args:
            frame: Current frame
            
        Returns:
            flow: Optical flow field (u, v components)
        """
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return None
        
        # 1. Convert to grayscale
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Handle size mismatch (resize current to match previous)
        if prev_gray.shape != curr_gray.shape:
            curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))
        
        # 3. Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **self.farneback_params
        )
        
        self.flow_history.append(flow)
        return flow
    
    def _analyze_progression(self):
        """
        Analyze flood progression from historical data
        
        Returns:
            stats: Dictionary containing progression statistics
        """
        if len(self.water_mask_history) < 2:
            return {
                "expansion_rate": 0.0,
                "direction": "unknown",
                "speed": 0.0,
                "predicted_expansion": 0.0
            }
        
        # Calculate water area change
        current_water = np.sum(self.water_mask_history[-1])
        previous_water = np.sum(self.water_mask_history[-2])
        
        if previous_water == 0:
            expansion_rate = 0.0
        else:
            expansion_rate = (current_water - previous_water) / previous_water
        
        # Analyze expansion direction
        direction = self._analyze_expansion_direction()
        
        # Calculate expansion speed (pixels per frame)
        speed = abs(current_water - previous_water)
        
        # Predict future expansion (simple linear extrapolation)
        predicted_expansion = self._predict_expansion()
        
        return {
            "expansion_rate": expansion_rate,
            "direction": direction,
            "speed": speed,
            "predicted_expansion": predicted_expansion
        }
    
    def _analyze_expansion_direction(self):
        """
        Analyze the primary direction of flood expansion
        
        Returns:
            direction: 'north', 'south', 'east', 'west', 'unknown'
        """
        if len(self.water_mask_history) < 2:
            return "unknown"
        
        # Calculate center of mass for water in consecutive frames
        def get_center_of_mass(mask):
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] == 0:
                return (mask.shape[1] // 2, mask.shape[0] // 2)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        
        current_center = get_center_of_mass(self.water_mask_history[-1])
        previous_center = get_center_of_mass(self.water_mask_history[-2])
        
        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            direction = 'east' if dx > 0 else 'west'
        else:
            direction = 'south' if dy > 0 else 'north'
        
        return direction
    
    def _predict_expansion(self):
        """
        Predict future flood expansion based on historical trend
        
        Returns:
            predicted_area: Predicted water area in next N frames
        """
        if len(self.water_mask_history) < 5:
            return 0.0
        
        # Calculate recent expansion rates
        recent_rates = []
        for i in range(len(self.water_mask_history) - 1, max(0, len(self.water_mask_history) - 6), -1):
            current = np.sum(self.water_mask_history[i])
            previous = np.sum(self.water_mask_history[i - 1])
            if previous > 0:
                rate = (current - previous) / previous
                recent_rates.append(rate)
        
        if not recent_rates:
            return 0.0
        
        # Average rate
        avg_rate = np.mean(recent_rates)
        
        # Predict next frame water area
        current_area = np.sum(self.water_mask_history[-1])
        predicted_area = current_area * (1 + avg_rate)
        
        return predicted_area
    
    def get_flood_progression_heatmap(self):
        """
        Generate heatmap showing flood progression over time
        
        Returns:
            heatmap: Visual heatmap of flood progression
        """
        if len(self.water_mask_history) < 2:
            return None
        
        # Combine historical masks with time-based weighting
        heatmap = np.zeros_like(self.water_mask_history[0], dtype=np.float32)
        
        for i, mask in enumerate(self.water_mask_history):
            # More recent masks get higher weight
            weight = (i + 1) / len(self.water_mask_history)
            heatmap += mask * weight
        
        # Normalize
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def get_temporal_risk_assessment(self):
        """
        Assess risk based on temporal flood behavior
        
        Returns:
            risk_level: 'LOW', 'MEDIUM', 'HIGH', 'DANGEROUS'
            risk_score: Numeric score (0-100)
            insights: List of risk insights
        """
        if len(self.water_mask_history) < 5:
            return "LOW", 0, ["Insufficient temporal data"]
        
        progression_stats = self._analyze_progression()
        
        insights = []
        score = 0
        
        # Rapid expansion
        if progression_stats['expansion_rate'] > 0.1:
            score += 40
            insights.append("Rapid flood expansion detected")
        
        # High speed
        if progression_stats['speed'] > 1000:
            score += 30
            insights.append("High flood spread speed")
        
        # Predicted significant expansion
        current_area = np.sum(self.water_mask_history[-1])
        if progression_stats['predicted_expansion'] > current_area * 1.5:
            score += 30
            insights.append("Flood expected to expand significantly")
        
        # Determine risk level
        if score < 20:
            level = "LOW"
        elif score < 50:
            level = "MEDIUM"
        elif score < 80:
            level = "HIGH"
        else:
            level = "DANGEROUS"
        
        if not insights:
            insights.append("Flood progression stable")
        
        return level, min(100, score), insights


if __name__ == "__main__":
    print("⏱️ Temporal Flood Tracker Module Ready")
    print("To use: tracker = TemporalFloodTracker()")
