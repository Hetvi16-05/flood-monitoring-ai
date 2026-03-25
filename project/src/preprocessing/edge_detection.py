import cv2

def canny_edge(image, low_threshold=100, high_threshold=200):
    """Apply Canny edge detection."""
    return cv2.Canny(image, low_threshold, high_threshold)

def sobel_edge(image, ksize=3):
    """Apply Sobel edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return cv2.magnitude(sobel_x, sobel_y)

def laplacian_edge(image, ksize=3):
    """Apply Laplacian edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
