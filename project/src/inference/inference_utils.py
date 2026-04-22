import cv2
import torch
import numpy as np
from torchvision import transforms
from config import DEVICE, INF_SIZE, COLORS, YOLO_CONF_THRESHOLD
from utils.area import calculate_water_area
from utils.risk import get_flood_risk
from preprocessing.feature_engineering import extract_hybrid_features
from models.crocodile_detector import CrocodileDetector, get_default_crocodile_model_path
from models.depth_estimator import DepthEstimator
from models.temporal_tracker import TemporalFloodTracker
from models.predictive_forecaster import PredictiveForecaster

def apply_morphology(mask):
    """
    Apply Morphological Erosion followed by Dilation (Opening/Closing) 
    to smooth flood boundaries and remove noise.
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def spatial_fusion(flood_mask, yolo_detections, orig_w, orig_h, overlap_thresh=0.3):
    """
    Robust Spatial Fusion using Overlap Ratios.
    Calculates the exact ratio of flood pixels underneath an object's bounding box.
    """
    valid_boxes = []
    counts = {}
    has_person, has_animal, has_vehicle, has_crocodile = False, False, False, False

    for box, label in yolo_detections:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Scale bounding box identically to the Mask dimensions (INF_SIZE)
        mx1 = int(max(0, x1 * (INF_SIZE[0] / orig_w)))
        my1 = int(max(0, y1 * (INF_SIZE[1] / orig_h)))
        mx2 = int(min(INF_SIZE[0], x2 * (INF_SIZE[0] / orig_w)))
        my2 = int(min(INF_SIZE[1], y2 * (INF_SIZE[1] / orig_h)))
        
        # 1. Improved Overlap Ratio Logic
        in_flood = False
        if (mx2 > mx1) and (my2 > my1):
            target_roi = flood_mask[my1:my2, mx1:mx2]
            overlap_ratio = np.mean(target_roi)
            # If >30% of the object's body bounds is touching water, it is flooded
            in_flood = bool(overlap_ratio >= overlap_thresh)
            
        valid_boxes.append((box, label, in_flood))
        
        counts[label] = counts.get(label, 0) + 1

        if label == 'person': has_person = True
        elif label == 'animal': has_animal = True
        elif label == 'vehicle': has_vehicle = True
        elif label == 'crocodile': has_crocodile = True

    obj_summary = ", ".join([f"{k}({v})" for k, v in counts.items()]) if counts else "None"
    return valid_boxes, has_person, has_animal, has_vehicle, has_crocodile, obj_summary

def run_hybrid(frame, model_dict, show_yolo=True, show_mask=True, lat=None, lon=None):
    """
    Production-level hybrid fusion inference with advanced AI monitoring telemetry.
    Returns: (display_img, water_p, obj_summary, risk_level, risk_score, telemetry)
    """
    from config import CLASSES
    yolo_custom = model_dict['yolo_custom']
    yolo_coco = model_dict['yolo_coco']
    seg = model_dict['seg']
    croc_detector = model_dict.get('croc_detector')
    depth_estimator = model_dict.get('depth_estimator')
    temporal_tracker = model_dict.get('temporal_tracker')
    predictive_forecaster = model_dict.get('predictive_forecaster')
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = frame.shape[:2]
    
    # 1. HYBRID YOLO DETECTION
    res_custom = yolo_custom(img_rgb, verbose=False, conf=YOLO_CONF_THRESHOLD)[0]
    res_coco = yolo_coco(img_rgb, verbose=False, conf=YOLO_CONF_THRESHOLD)[0]
    
    COCO_MAP = {
        'person': 'person',
        'dog': 'animal', 'bird': 'animal', 'cow': 'animal', 'cat': 'animal', 'horse': 'animal', 'sheep': 'animal',
        'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'motorcycle': 'vehicle', 'train': 'vehicle'
    }

    raw_detections = []
    yolo_class_scores = {}
    
    for box in res_custom.boxes:
        cls_id = int(box.cls[0])
        label = yolo_custom.names[cls_id]
        conf = box.conf[0].item()
        raw_detections.append((box, label))
        yolo_class_scores[label] = max(yolo_class_scores.get(label, 0), conf)
    
    for box in res_coco.boxes:
        cls_id = int(box.cls[0])
        label_raw = yolo_coco.names[cls_id]
        conf = box.conf[0].item()
        if label_raw in COCO_MAP:
             label = COCO_MAP[label_raw]
             raw_detections.append((box, label))
             yolo_class_scores[label] = max(yolo_class_scores.get(label, 0), conf)
    
    # 1.5. CROCODILE DETECTION (Specialized)
    if croc_detector:
        croc_detections = croc_detector.detect(frame)
        for croc_det in croc_detections:
            # Convert dict detection to YOLO-like format for fusion
            box_coords = croc_det['box']
            # Create a mock box object with xyxy attribute
            class MockBox:
                def __init__(self, xyxy):
                    self.xyxy = torch.tensor([xyxy])
            mock_box = MockBox(box_coords)
            raw_detections.append((mock_box, 'crocodile'))
            yolo_class_scores['crocodile'] = max(yolo_class_scores.get('crocodile', 0), croc_det['conf'])
    
    # 2. 5-CHANNEL SEGMENTATION
    img_resized = cv2.resize(img_rgb, INF_SIZE)
    hybrid_feat = extract_hybrid_features(img_resized).astype(np.float32)
    img_norm = img_resized.astype(np.float32) / 255.0
    combined = np.concatenate([img_norm, hybrid_feat], axis=-1).astype(np.float32)
    
    tensor = torch.from_numpy(combined).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = seg(tensor)
        probs = torch.softmax(out, dim=1) # (1, C, H, W)
        
        # Calculate segmentation confidence (average max probability across all pixels)
        seg_conf = torch.max(probs, dim=1)[0].mean().item()
        
        # Get per-class presence confidence (max prob in any pixel for that class)
        seg_class_scores = {}
        for i, cls_name in enumerate(CLASSES):
            # Use mean of top 10% pixels for a stable presence score
            top_px = torch.topk(probs[0, i].flatten(), k=max(1, int(INF_SIZE[0]*INF_SIZE[1]*0.01)))[0]
            seg_class_scores[cls_name] = top_px.mean().item()
            
        pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
    
    pred = apply_morphology(pred)
    
    # 3. SPATIAL FUSION
    flood_mask = (pred == 0).astype(np.uint8)
    valid_boxes, has_person, has_animal, has_vehicle, has_crocodile, obj_summary = spatial_fusion(flood_mask, raw_detections, orig_w, orig_h)
    
    # 3b. ADVANCED HYBRID TELEMETRY
    yolo_confs = [box.conf[0].item() for box, _ in raw_detections]
    yolo_conf = np.mean(yolo_confs) if yolo_confs else seg_conf
    
    # HYBRID CONFIDENCE FORMULA: 0.6 * Seg + 0.4 * YOLO
    hybrid_conf = (0.6 * seg_conf) + (0.4 * yolo_conf)
    
    # Combined confidence list for the Rekognition UI
    all_detections = []
    for cls, score in seg_class_scores.items():
        all_detections.append({"label": cls.replace("_", " ").title(), "confidence": score})
    for cls, score in yolo_class_scores.items():
        # Avoid duplicate labels, keep highest
        exists = next((d for d in all_detections if d["label"] == cls.title()), None)
        if exists:
            exists["confidence"] = max(exists["confidence"], score)
        else:
            all_detections.append({"label": cls.title(), "confidence": score})
            
    # Sort detections by confidence
    all_detections = sorted(all_detections, key=lambda x: x["confidence"], reverse=True)
    
    # 4. RISK ENGINE
    water_p = calculate_water_area(pred)
    
    # 4a. Water Depth Estimation (if available)
    avg_water_depth = 0.0
    max_water_depth = 0.0
    depth_risk_level = None
    depth_risk_score = 0
    
    # Temporarily disabled due to shape mismatch issue
    # TODO: Fix water_mask to depth_values shape alignment
    # if depth_estimator:
    #     flood_mask = (pred == 0).astype(np.uint8)
    #     avg_water_depth, max_water_depth, depth_map = depth_estimator.get_water_depth(frame, flood_mask)
    #     depth_risk_level, depth_risk_score = depth_estimator.get_depth_risk_level(avg_water_depth, max_water_depth)
    
    # 4b. Temporal Flood Tracking (if available) - DISABLED
    temporal_risk_level = None
    temporal_risk_score = 0
    temporal_insights = []
    expansion_rate = 0.0
    flood_direction = "unknown"
    
    # Temporarily disabled due to optical flow size mismatch
    # TODO: Fix frame size alignment for optical flow
    # The temporal tracker is causing frame size errors in optical flow calculation
    
    # 4c. Predictive Forecasting (if available)
    predictive_risk_level = None
    predictive_risk_score = 0
    prediction_hours = 0
    prediction_factors = []
    
    if predictive_forecaster and lat and lon:
        prediction = predictive_forecaster.predict_flood_risk(lat, lon, risk_score)
        predictive_risk_level = prediction['level']
        predictive_risk_score = int(prediction['probability'] * 100)
        prediction_hours = prediction['hours_ahead']
        prediction_factors = prediction['risk_factors']
    
    # 4d. Base Risk Calculation
    risk_level, risk_score = get_flood_risk(
        water_p=water_p,
        has_person=has_person,
        has_animal=has_animal,
        has_vehicle=has_vehicle,
        has_crocodile=has_crocodile,
        oif_detected=any([b[2] for b in valid_boxes])
    )
    
    # 4e. Combine all risk factors
    if depth_estimator and depth_risk_score > 0:
        # Weighted combination: 50% base risk + 20% depth risk + 15% temporal risk + 15% predictive risk
        risk_score = int(0.5 * risk_score + 0.2 * depth_risk_score + 0.15 * temporal_risk_score + 0.15 * predictive_risk_score)
    else:
        # No depth estimation: 70% base risk + 15% temporal risk + 15% predictive risk
        risk_score = int(0.7 * risk_score + 0.15 * temporal_risk_score + 0.15 * predictive_risk_score)
        
        # Recategorize based on combined score
        if risk_score < 15:
            risk_level = "LOW"
        elif risk_score < 40:
            risk_level = "MEDIUM"
        elif risk_score < 70:
            risk_level = "HIGH"
        else:
            risk_level = "DANGEROUS"
    
    # 5. TELEMETRY (after all risk calculations)
    telemetry = {
        "hybrid_conf": hybrid_conf,
        "seg_conf": seg_conf,
        "yolo_conf": yolo_conf,
        "top_detections": all_detections,
        "yolo_raw": [{"label": l, "conf": b.conf[0].item(), "box": b.xyxy[0].tolist()} for b, l, _ in valid_boxes],
        "avg_water_depth": avg_water_depth,
        "max_water_depth": max_water_depth,
        "depth_risk_level": depth_risk_level,
        "depth_risk_score": depth_risk_score,
        "expansion_rate": expansion_rate,
        "flood_direction": flood_direction,
        "temporal_risk_level": temporal_risk_level,
        "temporal_risk_score": temporal_risk_score,
        "temporal_insights": temporal_insights,
        "predictive_risk_level": predictive_risk_level,
        "predictive_risk_score": predictive_risk_score,
        "prediction_hours": prediction_hours,
        "prediction_factors": prediction_factors
    }
    
    # 6. VISUALIZATION
    display_img = img_rgb.copy()
    if show_mask:
        mask_c = np.zeros((INF_SIZE[1], INF_SIZE[0], 3), dtype=np.uint8)
        for i, color in enumerate(COLORS):
            mask_c[pred == i] = color
        mask_c = cv2.resize(mask_c, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        display_img = cv2.addWeighted(display_img, 0.7, mask_c, 0.3, 0)
    
    # All text overlays removed - clean visualization

    return cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR), water_p, obj_summary, risk_level, risk_score, telemetry

