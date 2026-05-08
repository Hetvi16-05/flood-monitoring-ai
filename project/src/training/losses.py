import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFloodLoss(nn.Module):
    """
    V3.1 High-Performance Loss: 70% Dice + 30% Cross-Entropy.
    Optimized for stable training on vision transformers.
    """
    def __init__(self, smooth=1.0):
        super(HybridFloodLoss, self).__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target):  
        # Apply softmax to get probabilities for each class
        pred = torch.softmax(pred, dim=1)
        pred_fg = pred[:, 1] # Foreground class (flood)

        # Ensure target is float for intersection
        target_fg = (target == 1).float()

        pred_fg = pred_fg.reshape(-1)
        target_fg = target_fg.reshape(-1)

        intersection = (pred_fg * target_fg).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_fg.sum() + target_fg.sum() + self.smooth)

        return 1 - dice

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce(pred, target)
        
        # Weighted combination for high-precision boundaries
        total = 0.7 * dice + 0.3 * ce
        return total
