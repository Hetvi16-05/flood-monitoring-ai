import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(inputs, targets, alpha=0.8, gamma=2.0):
    """
    Focal Loss to handle class imbalance (e.g., water vs background)
    """
    inputs = F.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE
    
    return focal_loss

def dice_loss(inputs, targets, smooth=1):
    """
    Dice Loss for segmentation overlap optimization
    """
    inputs = F.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    
    return 1 - dice

class WaterFocalDiceLoss(nn.Module):
    """
    Custom Loss for FloodNet: Focal + Dice + Boundary-Aware weight
    """
    def __init__(self, class_weights=None):
        super(WaterFocalDiceLoss, self).__init__()
        self.class_weights = class_weights if class_weights is not None else [3.0, 2.5, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0]

    def forward(self, pred, target):
        loss = 0
        for i in range(len(self.class_weights)):
            # Per-class loss
            p = pred[:, i]
            t = (target == i).float()
            
            f_loss = focal_loss(p, t)
            d_loss = dice_loss(p, t)
            
            loss += self.class_weights[i] * (f_loss + d_loss)
            
        return loss

class FloodDetectionLoss(nn.Module):
    """
    Submersion-aware detection loss
    """
    def __init__(self):
        super(FloodDetectionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # pred and target are dicts with 'bbox', 'classes', 'submersion'
        bbox_loss = F.smooth_l1_loss(pred['bbox'], target['bbox'])
        cls_loss = self.bce_loss(pred['classes'], target['classes'])
        submersion_loss = self.mse_loss(pred['submersion'], target['submersion'])
        
        return bbox_loss + cls_loss + 0.5 * submersion_loss

class RiskCalibrationLoss(nn.Module):
    """
    Mse + Calibration Error for Risk Engine
    """
    def __init__(self):
        super(RiskCalibrationLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_risk, true_risk):
        mse_loss = self.mse(pred_risk, true_risk)
        # Expected Calibration Error (simplified)
        # In practice, this requires binning probabilities
        calibration_loss = torch.mean(torch.abs(pred_risk - true_risk))
        return mse_loss + 0.2 * calibration_loss
