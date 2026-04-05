import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Important: Evaluate the 5-channel model
from models.segmentation_v2 import create_deeplabv3plus
from models.dataset import SegmentationDataset
from config import (
    DEVICE, NUM_CLASSES, CLASSES, SPLIT_DIR, OUTPUT_DIR, COLORS, WEIGHTS_DIR
)

def evaluate():
    eval_out = OUTPUT_DIR / "eval"
    os.makedirs(eval_out, exist_ok=True)
    
    # 1. Load V2 Model
    seg = create_deeplabv3plus(in_channels=5, num_classes=NUM_CLASSES)
    v2_weights = WEIGHTS_DIR / "best_model_v2.pth"
    if v2_weights.exists():
        seg.load_state_dict(torch.load(v2_weights, map_location=DEVICE))
        print("✅ Loaded V2 weights successfully.")
    else:
        print("⚠️ V2 weights not found. Using untrained weights.")
    # MUST ensure eval mode to avoid batchnorm errors with size 1 batch
    seg.eval()
    seg.to(DEVICE)
    
    val_img_dir = SPLIT_DIR / "val" / "images"
    val_mask_dir = SPLIT_DIR / "val" / "masks"
    
    # Pass transform=None to dataset, dataset natively adds 5-chan without albumentations
    val_dataset = SegmentationDataset(
        img_dir=str(val_img_dir),
        mask_dir=str(val_mask_dir),
        transform=None
    )
    
    # Needs a 5-channel compatible eval setup
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    class_iou = np.zeros(NUM_CLASSES)
    class_precision = np.zeros(NUM_CLASSES)
    class_recall = np.zeros(NUM_CLASSES)
    class_count = np.zeros(NUM_CLASSES)
    
    print(f"🚀 Starting 5-Channel Evaluation on {len(val_dataset)} images...")
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(val_loader)):
            image = image.to(DEVICE).float()
            mask = mask.to(DEVICE).long()
            
            output = seg(image)
            pred = torch.argmax(output, dim=1)[0].cpu().numpy()
            target = mask[0].cpu().numpy()
            
            mask_valid = target != 255
            all_preds.append(pred[mask_valid])
            all_targets.append(target[mask_valid])
            
            for cls in range(NUM_CLASSES):
                true_positive = np.logical_and(pred == cls, target == cls).sum()
                false_positive = np.logical_and(pred == cls, target != cls).sum()
                false_negative = np.logical_and(pred != cls, target == cls).sum()
                
                union = np.logical_or(pred == cls, target == cls).sum()
                
                if union > 0:
                    class_iou[cls] += true_positive / union
                    class_count[cls] += 1
                
                if (true_positive + false_positive) > 0:
                    class_precision[cls] += true_positive / (true_positive + false_positive)
                    
                if (true_positive + false_negative) > 0:
                    class_recall[cls] += true_positive / (true_positive + false_negative)
                    
            if i < 5:
                # Qualitative results
                # image has 5 channels. 0:3 is RGB. Reverse normalization to visualize
                img_vis = image[0, :3].cpu().permute(1, 2, 0).numpy()
                img_vis = img_vis * 255.0
                img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                
                gt_mask = np.zeros_like(img_vis)
                pr_mask = np.zeros_like(img_vis)
                for c_idx, color in enumerate(COLORS):
                    gt_mask[target == c_idx] = color
                    pr_mask[pred == c_idx] = color
                
                combined = np.hstack([img_vis, gt_mask, pr_mask])
                cv2.imwrite(str(eval_out / f"sample_v2_{i}.png"), combined)

    mean_iou, mean_precision, mean_recall = 0, 0, 0
    report = {}
    
    for cls in range(NUM_CLASSES):
        if class_count[cls] > 0:
            iou = class_iou[cls] / class_count[cls]
            prec = class_precision[cls] / class_count[cls]
            rec = class_recall[cls] / class_count[cls]
            
            report[CLASSES[cls]] = {"IoU": iou, "Precision": prec, "Recall": rec}
            mean_iou += iou
            mean_precision += prec
            mean_recall += rec
            
    mean_iou /= NUM_CLASSES
    mean_precision /= NUM_CLASSES
    mean_recall /= NUM_CLASSES

    # Display
    print("\n--- V2 PERFORMANCE SUMMARY ---")
    print(f"Mean IoU: {mean_iou:.4f} | Mean Precision: {mean_precision:.4f} | Mean Recall: {mean_recall:.4f}")
    for cls, metrics in report.items():
        print(f" - {cls}: IoU={metrics['IoU']:.4f}, Prec={metrics['Precision']:.4f}, Rec={metrics['Recall']:.4f}")
        
if __name__ == "__main__":
    evaluate()
