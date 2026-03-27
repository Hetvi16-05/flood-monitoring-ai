import cv2
import numpy as np

def calculate_water_area(mask: np.ndarray) -> float:
    """
    Calculate the percentage of water area in the mask with noise filtering.
    PHASE 3 Rules:
    - Morphology Open
    - Remove small components
    - Ignore tiny areas
    - connectedComponents
    """
    if mask is None or mask.size == 0:
        return 0.0
        
    # Create binary mask for water (class 0)
    water_mask = (mask == 0).astype(np.uint8)
    
    # 1. Morphological Opening (Remove isolated noise/pixels)
    # Using a 5x5 kernel as a balance between noise removal and keeping small genuine regions
    kernel = np.ones((5, 5), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    
    # 2. Connected Components Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(water_mask, connectivity=8)
    
    total_area = mask.size
    # Phase 3: ignore tiny areas, remove small components
    # We use 1% of total area as a threshold for "small/tiny"
    min_area_threshold = 0.01 * total_area  
    
    filtered_water_pixels = 0
    # Label 0 is background, start from 1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            filtered_water_pixels += area
            
    water_percent = (filtered_water_pixels / total_area) * 100
    return float(water_percent)
