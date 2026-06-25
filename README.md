# 🌧️ RAINWISE: Enterprise AI Computer Vision

RAINWISE is a high-performance, hybrid AI system for **Flood Monitoring and Urban Analysis**. It combines pixel-level semantic segmentation (LRASPP) with real-time object detection (YOLOv8) to provide a comprehensive situational awareness platform.This is mainly for Learning purpose.

## 🏗️ Architecture
The system follows a modular pipeline:
1.  **Input**: RGB Camera / Video / Image Folder.
2.  **Detection**: YOLOv8n (Magenta boxes for discrete objects).
3.  **Segmentation**: LRASPP MobileNetV3 (Color overlays for land cover).
4.  **Fusion**: Unified overlay with performance metrics.
5.  **Interface**: Streamlit Web UI & FastAPI Enterprise Service.

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Quick Run (CLI)
Use the unified `run.py` entry point:
- **Web Dashboard**: `python run.py --ui`
- **Real-time Camera**: `python run.py --camera`
- **Video Processing**: `python run.py --video path/to/video.mp4`
- **Batch Processing**: `python run.py --batch path/to/images/`
- **API Server**: `python run.py --api`

## 🧠 Model Details
- **Classes**: water, road, building, vegetation.
- **Hardware**: Fully optimized for Apple Silicon (MPS).
- **Backend**: PyTorch & Ultralytics.

## 🐳 Docker Deployment
```bash
docker-compose up --build
```
API will be available at `http://localhost:8000`.

---
Developed for high-stakes urban monitoring and disaster response.
