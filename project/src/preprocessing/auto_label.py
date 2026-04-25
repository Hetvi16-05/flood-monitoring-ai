import os
import cv2
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def auto_label_dataset(images_dir, model_path='yolov8n.pt', confidence=0.3):
    """
    Uses a pre-trained YOLO model to semi-automate labeling.
    Maps standard COCO classes to our flood classes if applicable.
    """
    model = YOLO(model_path)
    img_dir = Path(images_dir)
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    
    print(f"🤖 Auto-labeling {len(images)} images using {model_path}...")
    
    # Class mapping from COCO to RAINWISE
    # RAINWISE Classes: 0:water, 1:road, 2:building, 3:vehicle, 4:vegetation
    # COCO Classes: 2:car (vehicle), 5:bus (vehicle), 7:truck (vehicle), 9:boat (might be water?)
    coco_to_rainwise = {
        2: 3, 5: 3, 7: 3, # Vehicles
        9: 0             # Boat -> Water (rough heuristic)
    }

    for img_path in tqdm(images):
        # Verify image is readable
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Skipping corrupted image: {img_path}")
            continue

        try:
            results = model(img_path, stream=True, verbose=False)
            lbl_path = img_path.with_suffix('.txt')
            
            with open(lbl_path, 'w') as f:
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if cls_id in coco_to_rainwise:
                            new_cls = coco_to_rainwise[cls_id]
                            xywh = box.xywhn[0].tolist()
                            f.write(f"{new_cls} {' '.join(map(str, xywh))}\n")
        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")

    print(f"✅ Semi-auto labeling complete. Now open 'labelImg' to refine labels.")

if __name__ == "__main__":
    auto_label_dataset("/Users/HetviSheth/Flood_Prediction/project/raw_dataset/images")
