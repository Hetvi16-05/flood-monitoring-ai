import os
import sys

# Standardized Risk Levels
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
FLOOD = "FLOOD"
DANGER = "DANGER"

def get_flood_risk(water_percent: float, obj_count: int = 0, animal_count: int = 0) -> str:
    """
    Standardized risk assessment based on hybrid detection metrics.
    PHASE 2 & 5 Rules:
    - No objects detection = LOW (noise suppression)
    - water < 10 = LOW
    - water 10-30 = MEDIUM
    - water >30 and obj >1 = HIGH
    - water >50 and obj >2 = FLOOD
    - animal > 0 and water > 20 = DANGER
    """
    
    # 1. Noise Suppression Rule
    if obj_count == 0:
        return LOW
        
    # 2. Hierarchy of Risk (Highest to Lowest)
    
    # FLOOD: Extreme condition
    if water_percent >= 50 and obj_count > 2:
        return FLOOD
        
    # DANGER: Visual alert for animals in water (Phase 5)
    if animal_count > 0 and water_percent > 20:
        return DANGER
        
    # HIGH: Significant water with multiple objects
    if water_percent >= 30 and obj_count > 1:
        return HIGH
        
    # MEDIUM: Moderate water
    if water_percent >= 10:
        return MEDIUM
        
    # LOW: Default/Safe
    return LOW
