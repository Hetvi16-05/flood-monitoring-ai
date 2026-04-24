import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class WaterAttentionGate(nn.Module):
    """
    Attention gate that focuses on water-specific features like reflectivity (RGB)
    and index values (NDWI).
    """
    def __init__(self, in_channels):
        super(WaterAttentionGate, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn

class MultiScaleFusion(nn.Module):
    def __init__(self, high_res_channels, low_res_channels, out_channels):
        super(MultiScaleFusion, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(high_res_channels + low_res_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_res, low_res):
        low_res_up = self.up(low_res)
        combined = torch.cat([high_res, low_res_up], dim=1)
        return self.conv(combined)

class FloodNet(nn.Module):
    """
    FloodNet: Specialized Segmentation Model for Perfect Flood Detection.
    Features: 6-channel input, Water Attention Gates, Multi-scale Decoder.
    """
    def __init__(self, num_classes=8):
        super(FloodNet, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # 1. Modify Input Layer for 6-channels (RGB + NDWI + Texture + Elevation)
        self.initial_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize new channels with weight averages to preserve pre-training benefits
        with torch.no_grad():
            self.initial_conv.weight[:, :3] = resnet.conv1.weight
            self.initial_conv.weight[:, 3:] = resnet.conv1.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) / 3
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # 2. Encoder Blocks with Water Attention Gates
        self.layer1 = resnet.layer1
        self.attn1 = WaterAttentionGate(256)
        
        self.layer2 = resnet.layer2
        self.attn2 = WaterAttentionGate(512)
        
        self.layer3 = resnet.layer3
        self.attn3 = WaterAttentionGate(1024)
        
        self.layer4 = resnet.layer4
        self.attn4 = WaterAttentionGate(2048)
        
        # 3. Decoder with Multi-Scale Fusion & Skip Connections
        self.fusion3 = MultiScaleFusion(1024, 2048, 512)
        self.fusion2 = MultiScaleFusion(512, 512, 256)
        self.fusion1 = MultiScaleFusion(256, 256, 128)
        
        # 4. Final Output Head (8 classes)
        # Classes: flood_water, shallow_water, deep_water, debris, road, building, vegetation, emergency_objects
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x shape: [batch, 6, H, W]
        
        # Initial stages
        x = self.initial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_p = self.maxpool(x) # [batch, 64, H/4, W/4]
        
        # Encoder
        c1 = self.layer1(x_p) # [256, H/4, W/4]
        c1 = self.attn1(c1)
        
        c2 = self.layer2(c1)  # [512, H/8, W/8]
        c2 = self.attn2(c2)
        
        c3 = self.layer3(c2)  # [1024, H/16, W/16]
        c3 = self.attn3(c3)
        
        c4 = self.layer4(c3)  # [2048, H/32, W/32]
        c4 = self.attn4(c4)
        
        # Decoder
        d3 = self.fusion3(c3, c4) # [512, H/16, W/16]
        d2 = self.fusion2(c2, d3) # [256, H/8, W/8]
        d1 = self.fusion1(c1, d2) # [128, H/4, W/4]
        
        # Output
        out = self.final_up(d1)    # [128, H, W]
        out = self.classifier(out) # [8, H, W]
        
        return out

if __name__ == "__main__":
    # Quick sanity check
    model = FloodNet(num_classes=8)
    dummy_input = torch.randn(1, 6, 512, 512)
    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") # Expect [1, 8, 512, 512]
