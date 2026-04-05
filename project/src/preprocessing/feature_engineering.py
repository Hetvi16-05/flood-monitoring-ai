import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def get_canny_features(image, low_threshold=50, high_threshold=150):
    """
    Extract Canny edges to help segmentation with boundary detection.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    # Normalize to 0-1
    return edges.astype(np.float32) / 255.0

def get_lbp_features(image, radius=3, n_points=24):
    """
    Extract LBP (Local Binary Pattern) features for texture analysis.
    Helping distinguish between 'road' (smooth) and 'vegetation' (rough).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    # Normalize to 0-1
    return lbp.astype(np.float32) / np.max(lbp)

def extract_hybrid_features(image):
    """
    Extract multiple spatial/texture features and return as a 2-channel stack.
    """
    canny = get_canny_features(image)
    lbp = get_lbp_features(image)
    
    # (H, W, 2)
    return np.stack([canny, lbp], axis=-1)

if __name__ == "__main__":
    print("🚀 Feature Engineering Module: Canny + LBP Handlers Active.")
