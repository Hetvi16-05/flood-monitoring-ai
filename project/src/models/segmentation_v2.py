import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
from config import DEVICE, NUM_CLASSES

def create_deeplabv3plus(in_channels=5, num_classes=6):
    """
    Industry-level segmentation model (DeepLabV3+).
    Supports multi-channel input (RGB + Edge + Texture).
    """
    # 1. Create a 3-channel model to get pretrained weights
    temp_model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )
    pretrained_weight = temp_model.encoder.conv1.weight.data.clone()
    
    # 2. Create the target 5-channel model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None, # We'll manually load
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    
    # 3. Load adjusted weights
    new_weight = model.encoder.conv1.weight.data.clone()
    new_weight[:, :3, :, :] = pretrained_weight
    # Initialize channels 4 and 5 (Edge/Texture) with the mean of RGB weights
    avg_weight = pretrained_weight.mean(dim=1, keepdim=True)
    new_weight[:, 3:4, :, :] = avg_weight
    new_weight[:, 4:5, :, :] = avg_weight
    
    model.encoder.conv1.weight = nn.Parameter(new_weight)
    
    return model.to(DEVICE)

if __name__ == "__main__":
    model = create_deeplabv3plus(in_channels=5, num_classes=6)
    model.eval()
    dummy_input = torch.randn(1, 5, 256, 256).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"🚀 Model V2 Initialized. Input: {dummy_input.shape} -> Output: {output.shape}")
