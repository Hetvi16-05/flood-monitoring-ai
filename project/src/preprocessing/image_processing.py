import cv2
import numpy as np

def resize_image(image, size=(256, 256)):
    """Resize image to target size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def normalize_image(image):
    """Normalize image pixels to [0, 1] range."""
    return image.astype(np.float32) / 255.0

def convert_to_rgb(image):
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def histogram_equalization(image):
    """Apply histogram equalization to improve contrast."""
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    else:
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
