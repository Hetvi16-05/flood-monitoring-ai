import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
import evaluate
import os
from pathlib import Path

def train_hf_flood_model():
    print("🚀 FINE-TUNING HUGGING FACE FLOOD CLASSIFIER (SigLIP)")
    print("-----------------------------------------------------")
    
    model_id = "prithivMLmods/Flood-Image-Detection"
    output_dir = "newmodel/hf_training_runs"
    
    # 1. Load Image Processor and Model
    # SigLIP doesn't use a tokenizer, so we use AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    
    # 2. Data Transforms
    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load Datasets
    raw_train_ds = ImageFolder("newmodel/classification_dataset/train", transform=transforms)
    raw_val_ds = ImageFolder("newmodel/classification_dataset/valid", transform=transforms)
    
    # Wrap datasets to return dicts instead of tuples (Fixes vars() error)
    class HFDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            img, lbl = self.ds[i]
            return {"pixel_values": img, "label": lbl}
            
    train_ds = HFDatasetWrapper(raw_train_ds)
    val_ds = HFDatasetWrapper(raw_val_ds)
    
    print(f"📊 Training on {len(train_ds)} images across 2 classes: {raw_train_ds.classes}")
    
    # 4. Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch", # RENAME
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=10, 
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    
    # 7. Resume Logic
    last_checkpoint = None
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        last_checkpoint = True # Trainer will automatically find the latest
        print(f"♻️ RESUMING: Found existing checkpoints in {output_dir}")
    
    # 8. Start Training
    print("\n🎬 Training starting on MPS (Apple Silicon)...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 9. Save Final Model
    trainer.save_model("newmodel/hf_flood_model_final")
    print("\n✅ SUCCESS: Transformer Model saved in newmodel/hf_flood_model_final")

if __name__ == "__main__":
    train_hf_flood_model()
