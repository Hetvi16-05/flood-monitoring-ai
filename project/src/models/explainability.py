import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class FloodGradCAM:
    """
    Grad-CAM implementation for RAINWISE Flood Models.
    Highlights regions in the input image that contribute most to a specific risk prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients and activations
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=0):
        """
        Generates a Grad-CAM heatmap for a given input and class.
        Args:
            input_tensor: [1, C, H, W]
            class_idx: Index of the class (e.g., 0 for 'flood_water')
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # If output is [1, C, H, W] (segmentation), we take the mean of the target class
        score = output[:, class_idx].mean()
        score.backward()
        
        # Pooling gradients
        gradients = self.gradients # [1, C, H', W']
        activations = self.activations # [1, C, H', W']
        
        b, c, h, w = gradients.shape
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1).squeeze()
        
        # ReLU to keep only positive influence
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.detach().cpu().numpy()
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-10)
        
        # Resize to input resolution
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        
        return cam

def overlay_heatmap(img, heatmap, alpha=0.5):
    """
    Overlays a heatmap on an image.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # img should be [H, W, 3] in 0-255
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay

if __name__ == "__main__":
    # Test logic
    from models.flood_transformer import SwinFloodNet
    device = "cpu"
    model = SwinFloodNet(num_classes=8, in_channels=6).to(device)
    
    # Target the last feature map of the backbone
    target_layer = model.backbone.feature_info[-1]['module']
    
    gcam = FloodGradCAM(model, target_layer)
    dummy_input = torch.randn(1, 6, 256, 256).to(device)
    heatmap = gcam.generate_heatmap(dummy_input)
    print(f"Heatmap generated: {heatmap.shape}")
