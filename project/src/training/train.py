import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from pathlib import Path

# -----------------------------
# PATH FIX
# -----------------------------
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from project.src.models.dataset import SegmentationDataset
from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    WEIGHTS_DIR,
)

# -----------------------------
# Dice Loss
# -----------------------------
class DiceLoss(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target):

        smooth = 1e-6
        
        # Mask out 255 (ignore index)
        valid_mask = (target != 255).unsqueeze(1).float()
        
        inputs = F.softmax(inputs, dim=1)
        
        # Temporarily map 255 to 0 for one_hot, then we'll mask it
        target_fixed = target.clone()
        target_fixed[target == 255] = 0

        target_one_hot = F.one_hot(
            target_fixed,
            self.n_classes
        ).permute(0, 3, 1, 2).float()

        # Apply mask
        inputs = inputs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        intersection = torch.sum(
            inputs * target_one_hot,
            (2, 3)
        )

        union = torch.sum(
            inputs + target_one_hot,
            (2, 3)
        )

        dice = (2 * intersection + smooth) / (union + smooth)

        return 1 - dice.mean()


# -----------------------------
# IoU
# -----------------------------
def compute_iou(pred, target, num_classes):

    pred = torch.argmax(pred, dim=1)
    
    # Mask out 255 (ignore index)
    valid_mask = (target != 255)

    iou = 0
    count = 0

    for cls in range(num_classes):

        p = (pred == cls) & valid_mask
        t = (target == cls) & valid_mask

        inter = (p & t).sum().float()
        union = (p | t).sum().float()

        if union == 0:
            continue

        iou += inter / union
        count += 1

    if count == 0:
        return torch.tensor(0.0)

    return iou / count


# -----------------------------
# TRAIN
# -----------------------------
def train_model():

    # -----------------------------
    # DEVICE
    # -----------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("🚀 Device:", device)

    # -----------------------------
    # TRANSFORMS (FIXED)
    # -----------------------------
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        normalize
    ])

    val_transform = transforms.Compose([
        normalize
    ])

    # -----------------------------
    # DATASET
    # -----------------------------
    full_dataset = SegmentationDataset()

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_idx, val_idx = torch.utils.data.random_split(
        range(len(full_dataset)),
        [train_size, val_size]
    )

    train_dataset = torch.utils.data.Subset(
        SegmentationDataset(transform=train_transform),
        train_idx
    )

    val_dataset = torch.utils.data.Subset(
        SegmentationDataset(transform=val_transform),
        val_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # -----------------------------
    # MODEL (FAST + STRONG)
    # -----------------------------
    model = models.segmentation.lraspp_mobilenet_v3_large(
        weights="DEFAULT"
    )

    model.classifier.low_classifier = nn.Conv2d(
        40,
        NUM_CLASSES,
        1
    )

    model.classifier.high_classifier = nn.Conv2d(
        128,
        NUM_CLASSES,
        1
    )

    model.to(device)

    # -----------------------------
    # LOSS
    # -----------------------------
    # water: 1.0, road: 1.5, building: 2.0, vegetation: 1.2
    weights = torch.tensor([1.0, 1.5, 2.0, 1.2]).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    dice_loss = DiceLoss(NUM_CLASSES)

    # -----------------------------
    # OPTIMIZER
    # -----------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.5
    )

    # -----------------------------
    # SETTINGS
    # -----------------------------
    MAX_STEPS = 1000
    best_loss = 999
    patience = 5
    no_improve = 0

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # -----------------------------
    # LOOP
    # -----------------------------
    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0
        steps = 0

        for i, (images, masks) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}")
        ):

            if i > MAX_STEPS:
                break

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            out = model(images)["out"]

            loss = (
                0.4 * ce_loss(out, masks)
                + 0.6 * dice_loss(out, masks)
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            steps += 1

        train_loss /= steps

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()

        val_loss = 0
        val_iou = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                out = model(images)["out"]

                loss = (
                    0.4 * ce_loss(out, masks)
                    + 0.6 * dice_loss(out, masks)
                )

                val_loss += loss.item()

                val_iou += compute_iou(
                    out,
                    masks,
                    NUM_CLASSES
                ).item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_loss)

        print(
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"IoU {val_iou:.4f}"
        )

        # -----------------------------
        # SAVE BEST
        # -----------------------------
        if val_loss < best_loss:

            best_loss = val_loss
            no_improve = 0

            torch.save(
                model.state_dict(),
                os.path.join(
                    WEIGHTS_DIR,
                    "best_model.pth"
                )
            )

            print("✅ Best saved")

        else:

            no_improve += 1

            if no_improve >= patience:
                print("🛑 Early stop")
                break


if __name__ == "__main__":
    train_model()