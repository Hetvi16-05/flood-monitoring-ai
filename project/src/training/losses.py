import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    """
    V3.1 Specialized Loss: 50% Dice + 50% Focal.
    Perfect for hard environmental samples (reflections, muddy water).
    """
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def dice_loss(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        return 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        total_loss = 0
        
        # We iterate through foreground classes
        for i in range(num_classes):
            p = torch.sigmoid(pred[:, i])
            t = (target == i).float()
            
            # 1. Focal Loss Component
            bce = F.binary_cross_entropy(p, t, reduction='none')
            p_t = p * t + (1 - p) * (1 - t)
            f_loss = self.alpha * (1 - p_t)**self.gamma * bce
            f_loss = f_loss.mean()
            
            # 2. Dice Loss Component
            d_loss = self.dice_loss(pred[:, i], t, self.smooth)
            
            total_loss += (0.5 * f_loss + 0.5 * d_loss)
            
        return total_loss
