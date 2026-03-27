import cv2
import sys
import os
from pathlib import Path
from .inference_utils import run_hybrid
from ..utils.logger import get_logger

logger = get_logger("CCTVStream")

class CCTVStream:
    def __init__(self, url):
        """
        Initialize CCTV Stream handler.
        Supports: RTSP, HTTP, IP Camera URLs or 0 (Webcam).
        """
        self.url = url
        # Handle numeric string for webcam (e.g. "0")
        if isinstance(url, str) and url.isdigit():
            self.url = int(url)
            
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            logger.error(f"Failed to open CCTV stream: {url}")
            raise ConnectionError(f"Could not connect to stream: {url}")
        
        logger.info(f"Stream opened successfully: {url}")

    def read(self, yolo, seg):
        """
        Read a frame from the stream and run hybrid inference.
        Returns: frame_out, water_p, objects, animals, risk
        """
        if not self.cap.isOpened():
            return None, 0, [], [], "OFFLINE"

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to receive frame from stream.")
            return None, 0, [], [], "OFFLINE"

        try:
            # Run the standardized hybrid inference
            frame_out, water_p, objects, animals, risk = run_hybrid(frame, yolo, seg)
            return frame_out, water_p, objects, animals, risk
        except Exception as e:
            logger.error(f"Inference error in CCTV stream: {e}")
            return frame, 0, [], [], "ERROR"

    def release(self):
        """Release the video capture resource."""
        if self.cap.isOpened():
            self.cap.release()
            logger.info("CCTV Stream released.")
