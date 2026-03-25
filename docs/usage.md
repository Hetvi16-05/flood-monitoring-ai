# RAINWISE Usage Guide

## 🖥️ Command Line Interface (CLI)
The `run.py` script is the primary entry point.

### Object Detection & Segmentation
- **Single Image**: `python run.py --image test.jpg`
- **Live Stream**: `python run.py --camera`
- **Video File**: `python run.py --video input.mp4`

## 🌍 Web & API Services
- **Streamlit**: Launch a local web app for drag-and-drop analysis.
- **FastAPI**: Integrate RAINWISE into other services via HTTP REST.

## 📊 Deployment
The system can be deployed as a Docker container for cloud environments or run locally on Mac/Windows/Linux with GPU acceleration.
