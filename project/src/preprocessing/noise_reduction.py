import cv2

def gaussian_noise_remove(image, kernel_size=(5, 5)):
    """Apply Gaussian Blur to remove noise."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def median_filter(image, kernel_size=5):
    """Apply Median Blur to remove impulse noise."""
    return cv2.medianBlur(image, kernel_size)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply Bilateral Filter to remove noise while preserving edges."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
