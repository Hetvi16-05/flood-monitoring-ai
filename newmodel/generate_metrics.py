import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluate_and_plot():
    print("📊 GENERATING MASTER CONFUSION MATRIX")
    print("--------------------------------------")
    
    model_path = "newmodel/hf_flood_model_final"
    if not os.path.exists(model_path):
        # Fallback to latest checkpoint if final isn't saved yet
        checkpoints = [d for d in os.listdir("newmodel/hf_training_runs") if d.startswith("checkpoint")]
        if not checkpoints:
            print("❌ No model found! Please wait for at least one epoch to finish.")
            return
        model_path = os.path.join("newmodel/hf_training_runs", sorted(checkpoints)[-1])
    
    print(f"🎯 Loading Model from: {model_path}")
    
    # 1. Load Processor from original ID and Model from checkpoint
    # This fixes the OSError for missing config files in checkpoints
    original_id = "prithivMLmods/Flood-Image-Detection"
    processor = AutoImageProcessor.from_pretrained(original_id)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.eval()
    if torch.backends.mps.is_available(): model.to("mps")
    
    # 2. Data Transforms
    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load Validation Set
    val_ds = ImageFolder("newmodel/classification_dataset/valid", transform=transforms)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print(f"🎬 Running inference on {len(val_ds)} images...")
    
    with torch.no_grad():
        for imgs, lbls in val_loader:
            if torch.backends.mps.is_available(): imgs = imgs.to("mps")
            outputs = model(imgs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
            
    # 4. Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=val_ds.classes)
    
    print("\n✅ CLASSIFICATION REPORT:")
    print(cr)
    
    # 5. Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=val_ds.classes, yticklabels=val_ds.classes)
    plt.xlabel('PREDICTED')
    plt.ylabel('ACTUAL')
    plt.title('RAINWISE V3.1 - Master Confusion Matrix')
    
    output_img = "newmodel/confusion_matrix.png"
    plt.savefig(output_img)
    print(f"\n🖼️ Confusion Matrix saved as: {output_img}")
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()
