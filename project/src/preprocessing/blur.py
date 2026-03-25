import cv2
import numpy as np

def gaussian_blur(image, kernel_size=(5, 5)):
    """Apply Gaussian Blur."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def median_blur(image, kernel_size=5):
    """Apply Median Blur."""
    return cv2.medianBlur(image, kernel_size)

def motion_blur(image, kernel_size=15):
    """Apply Motion Blur to an image."""
    kernel_v = np.zeros((kernel_size, kernel_size))
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_v /= kernel_size
    return cv2.filter2D(image, -1, kernel_v)
