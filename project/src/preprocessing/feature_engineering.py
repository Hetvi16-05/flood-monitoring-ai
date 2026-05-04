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
    return edges.astype(np.float32) / 255.0

def get_lbp_features(image, radius=3, n_points=24):
    """
    Extract LBP features for texture analysis.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    return lbp.astype(np.float32) / (np.max(lbp) + 1e-6)

def get_ndwi_proxy(image):
    """
    Extract a proxy for NDWI using RGB channels.
    """
    # image is RGB
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    ndwi = (G - R) / (G + R + 1e-6)
    return (ndwi + 1) / 2

def get_sobel_texture(image):
    """
    Extract Sobel texture (matches FloodCustomDataset).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    texture = np.sqrt(sobelx**2 + sobely**2)
    return cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

def get_elevation_mock(shape):
    """
    Generate mock elevation gradient (matches FloodCustomDataset).
    """
    h, w = shape[:2]
    return np.linspace(1, 0, h).reshape(h, 1).repeat(w, axis=1).astype(np.float32)

def extract_hybrid_features(image, channels=2):
    """
    Extract multiple features.
    channels=2: Canny + LBP (Legacy V2)
    channels=3: NDWI + Sobel Texture + Elevation (Transformer V3)
    """
    if channels == 3:
        ndwi = get_ndwi_proxy(image)
        texture = get_sobel_texture(image)
        elevation = get_elevation_mock(image.shape)
        return np.stack([ndwi, texture, elevation], axis=-1)
    
    # Default 2 channels (Legacy)
    canny = get_canny_features(image)
    lbp = get_lbp_features(image)
    return np.stack([canny, lbp], axis=-1)
