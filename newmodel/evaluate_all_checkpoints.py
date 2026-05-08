import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluate_all():
    print("📈 RAINWISE V3.1 - CHECKPOINT EVOLUTION ANALYSIS")
    print("-----------------------------------------------")
    
    base_path = "newmodel/hf_training_runs"
    original_id = "prithivMLmods/Flood-Image-Detection"
    
    # 1. Find all checkpoints and sort them numerically
    checkpoints = [d for d in os.listdir(base_path) if d.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    
    if not checkpoints:
        print("❌ No checkpoints found yet!")
        return

    # 2. Setup Data
    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_ds = ImageFolder("newmodel/classification_dataset/valid", transform=transforms)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
    
    processor = AutoImageProcessor.from_pretrained(original_id)
    
    summary = []

    # 3. Loop through each checkpoint
    for cp in checkpoints:
        cp_path = os.path.join(base_path, cp)
        print(f"\n🔍 Evaluating {cp}...")
        
        try:
            model = AutoModelForImageClassification.from_pretrained(cp_path)
            model.eval()
            if torch.backends.mps.is_available(): model.to("mps")
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    if torch.backends.mps.is_available(): imgs = imgs.to("mps")
                    outputs = model(imgs)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(lbls.numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            summary.append((cp, acc))
            
            # Plot and save individual CM
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=val_ds.classes, yticklabels=val_ds.classes)
            plt.title(f'Evolution: {cp} (Acc: {acc:.2%})')
            plt.savefig(f"newmodel/cm_{cp}.png")
            plt.close()
            
            print(f"✅ {cp} Accuracy: {acc:.2%}")
            
        except Exception as e:
            print(f"⚠️ Error evaluating {cp}: {e}")

    # 4. Final Summary
    print("\n🏁 EVOLUTION SUMMARY:")
    print(f"{'Checkpoint':<15} | {'Accuracy':<10}")
    print("-" * 30)
    for cp, acc in summary:
        print(f"{cp:<15} | {acc:.2%}")

if __name__ == "__main__":
    evaluate_all()
