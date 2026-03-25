# 📈 Model Benchmark Report

## 🏥 Hardware Stats
- **Device**: Apple Silicon (MPS) / CPU Fallback
- **Inference Size**: 640 x 640

## 🎯 Model Performance
- **Segmentation (mIoU)**: ~0.7179
- **Detection (mAP50)**: Standard YOLOv8n COCO
- **Latency (Avg)**:
    - LRASPP: ~30-50ms
    - YOLOv8: ~15-20ms
    - Total: ~50-70ms per frame

## 📦 Dataset Info
- **Source**: RAINWISE Cleaned Dataset
- **Pre-processing**: Resize (256x256 Train), Normalize, Mask Refinement.
- **Classes**: 4 (Water, Road, Building, Vegetation).

## 🚀 Optimization
- `torch.no_grad()` active for all inference.
- Non-blocking logging.
- Memory-mapped weights loading.
