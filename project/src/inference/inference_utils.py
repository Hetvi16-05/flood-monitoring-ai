import cv2
import torch
import numpy as np
from torchvision import transforms
from ..config import DEVICE, INF_SIZE, COLORS, YOLO_CONF_THRESHOLD
from ..utils.area import calculate_water_area
from ..utils.risk import get_flood_risk

# Transform for segmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(INF_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_hybrid(frame, yolo, seg, show_yolo=True, show_mask=True):
    """
    Standardized hybrid inference function.
    Returns: frame_out, water_p, objects, animals, risk
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = frame.shape[:2]
    
    # 1. YOLO Detection
    yolo_results = yolo(img_rgb, verbose=False)[0]
    
    # Filter for objects and animals
    valid_boxes = []
    animals = []
    objects = []
    
    # Animal categories (dog, cow, horse, snake, crocodile)
    # Note: official YOLOv8n has dog(16), horse(17), cow(19). 
    # Crocodile and Snake might not be in VOC/COCO, but we'll try to detect common ones.
    ANIMAL_IDS = [16, 17, 19] # Dog, Horse, Cow
    
    for box in yolo_results.boxes:
        conf = float(box.conf[0])
        if conf > YOLO_CONF_THRESHOLD:
            cls_id = int(box.cls[0])
            label = yolo.names[cls_id]
            valid_boxes.append(box)
            if cls_id in ANIMAL_IDS:
                animals.append(label)
            else:
                objects.append(label)

    obj_count = len(valid_boxes)
    animal_count = len(animals)
    
    # 2. Segmentation
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = seg(tensor)["out"]
        pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
    
    # 3. Analytics
    water_p = calculate_water_area(pred)
    risk = get_flood_risk(water_p, obj_count, animal_count)
    
    # 4. Layered Overlay
    display_img = img_rgb.copy()
    
    # Layer 2: Mask (Transparent Alpha 0.3)
    # Only if objects detected (Phase 4 rule)
    if obj_count > 0 and show_mask:
        mask_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for i, color in enumerate(COLORS):
            mask_colored[pred == i] = color
        mask_colored = cv2.resize(mask_colored, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        display_img = cv2.addWeighted(display_img, 0.7, mask_colored, 0.3, 0)
    
    # Layer 3 & 4: Boxes and Text
    if show_yolo:
        for box in valid_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo.names[int(box.cls[0])]
            conf = float(box.conf[0])
            color = (255, 0, 255) if label in animals else (0, 255, 0)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Convert back to BGR for standard processing
    frame_out = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    
    return frame_out, water_p, list(set(objects)), list(set(animals)), risk
