# RAINWISE System Architecture

## 🧬 Data Flow Pipeline
1.  **Ingestion**: Images/Frames are resized to 640x640 (Inference Size).
2.  **Detection (YOLOv8n)**: Identifies discrete objects like cars, people, or signs.
3.  **Segmentation (LRASPP)**: Classifies every pixel into 4 categories:
    - 🔵 Water
    - 🟢 Road
    - 🔴 Building
    - 🟠 Vegetation
4.  **Overlay Engine**: Combines binary masks and bounding boxes with original image.
5.  **Metrics Handler**: Calculates latency per model and total FPS.

## 📁 Modules
- `project/src/api/`: FastAPI endpoints.
- `project/src/inference/`: Hybrid pipelines (Cam, Video, Batch).
- `project/src/ui/`: Streamlit frontend.
- `project/src/utils/`: Logger and metrics.
