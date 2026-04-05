import cv2
import torch
import numpy as np
from torchvision import transforms
from config import DEVICE, INF_SIZE, COLORS, YOLO_CONF_THRESHOLD
from utils.area import calculate_water_area
from utils.risk import get_flood_risk
from preprocessing.feature_engineering import extract_hybrid_features

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
    has_person, has_animal, has_vehicle = False, False, False

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

    obj_summary = ", ".join([f"{k}({v})" for k, v in counts.items()]) if counts else "None"
    return valid_boxes, has_person, has_animal, has_vehicle, obj_summary

def run_hybrid(frame, model_dict, show_yolo=True, show_mask=True):
    """
    Production-level hybrid fusion inference.
    """
    yolo_custom = model_dict['yolo_custom']
    yolo_coco = model_dict['yolo_coco']
    seg = model_dict['seg']
    
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
    for box in res_custom.boxes:
        cls_id = int(box.cls[0])
        label = yolo_custom.names[cls_id]
        raw_detections.append((box, label))
    
    for box in res_coco.boxes:
        cls_id = int(box.cls[0])
        label_raw = yolo_coco.names[cls_id]
        if label_raw in COCO_MAP:
             raw_detections.append((box, COCO_MAP[label_raw]))
    
    # 2. 5-CHANNEL SEGMENTATION
    img_resized = cv2.resize(img_rgb, INF_SIZE)
    hybrid_feat = extract_hybrid_features(img_resized)
    img_norm = img_resized.astype(np.float32) / 255.0
    combined = np.concatenate([img_norm, hybrid_feat], axis=-1)
    
    tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = seg(tensor)
        # Calculate segmentation confidence
        probs = torch.softmax(out, dim=1)
        seg_conf = torch.max(probs, dim=1)[0].mean().item()
        pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
    
    pred = apply_morphology(pred)
    
    # 3. SPATIAL FUSION (Check If Object in Flood)
    flood_mask = (pred == 0).astype(np.uint8)
    valid_boxes, has_person, has_animal, has_vehicle, obj_summary = spatial_fusion(flood_mask, raw_detections, orig_w, orig_h)
    
    # 3b. Calculate Hybrid Confidence
    yolo_confs = [box.conf[0].item() for box, _ in raw_detections]
    yolo_conf = np.mean(yolo_confs) if yolo_confs else seg_conf
    final_confidence = (seg_conf + yolo_conf) / 2
    
    # 4. ADVANCED RISK ENGINE
    water_p = calculate_water_area(pred)
    risk_level, risk_score = get_flood_risk(
        water_p=water_p,
        has_person=has_person,
        has_animal=has_animal,
        has_vehicle=has_vehicle,
        oif_detected=any([b[2] for b in valid_boxes]) # OIF: Object-in-Flood
    )
    
    # 5. VISUALIZATION
    display_img = img_rgb.copy()
    if show_mask:
        mask_c = np.zeros((INF_SIZE[1], INF_SIZE[0], 3), dtype=np.uint8)
        for i, color in enumerate(COLORS):
            mask_c[pred == i] = color
        mask_c = cv2.resize(mask_c, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        display_img = cv2.addWeighted(display_img, 0.7, mask_c, 0.3, 0)
    
    # Top Panel
    cv2.rectangle(display_img, (0, 0), (orig_w, 75), (0,0,0), -1)
    status_text = f"WATER: {water_p:.1f}% | RISK: {risk_level} ({risk_score}/100) | CONF: {final_confidence:.1%}"
    cv2.putText(display_img, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(display_img, f"OBJECTS: {obj_summary}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

    if show_yolo:
        for box, label, in_flood in valid_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 0, 255) if in_flood else (0, 255, 0) # Red if inside water
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_img, f"{label} ({box.conf[0]:.2f}) {'(FLOOD)' if in_flood else ''}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    return cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR), water_p, obj_summary, risk_level, risk_score, final_confidence
