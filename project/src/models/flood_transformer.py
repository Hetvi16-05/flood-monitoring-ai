import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List

class MLP(nn.Module):
    """
    Lightweight MLP layer for feature projection.
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # Handle both [B, C, H, W] and [B, H, W, C]
        if x.dim() == 4:
            if x.shape[-1] == self.proj.in_features:
                B, H, W, C = x.shape
                x = x.permute(0, 3, 1, 2).contiguous() # To [B, C, H, W]
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous() # [B, N, C]
        x = self.proj(x)
        x = x.transpose(1, 2).contiguous().reshape(B, -1, H, W)
        return x

class SwinFloodNet(nn.Module):
    """
    SwinFloodNet: A Transformer-based Segmentation Model for Flood Detection.
    Backbone: Swin Transformer V2 (Tiny)
    Decoder: Multi-scale MLP Fusion (SegFormer-style)
    """
    def __init__(self, num_classes=8, in_channels=6, embed_dim=256):
        super().__init__()
        
        # 1. Initialize Swin-V2 Backbone
        self.backbone = timm.create_model(
            'swinv2_tiny_window8_256', 
            pretrained=True, 
            in_chans=3, 
            features_only=True
        )
        
        # 2. Modify Patch Embedding for 6-channel input
        old_proj = self.backbone.patch_embed.proj
        new_proj = nn.Conv2d(
            in_channels, 
            old_proj.out_channels, 
            kernel_size=old_proj.kernel_size, 
            stride=old_proj.stride, 
            padding=old_proj.padding
        )
        
        with torch.no_grad():
            new_proj.weight[:, :3] = old_proj.weight
            new_proj.weight[:, 3:] = old_proj.weight.mean(dim=1, keepdim=True).repeat(1, in_channels-3, 1, 1) / (in_channels-3)
        
        self.backbone.patch_embed.proj = new_proj
        
        # 3. MLP Decoder Layers
        # Swin Tiny feature dims: [96, 192, 384, 768]
        dims = [96, 192, 384, 768]
        self.linear_c4 = MLP(input_dim=dims[3], embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=dims[2], embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=dims[1], embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=dims[0], embed_dim=embed_dim)
        
        # 4. Fusion Layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 5. Final Classifier Head
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Extract features from backbone
        features = self.backbone(x)
        c1, c2, c3, c4 = features
        
        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)
        _c1 = self.linear_c1(c1)
        
        target_size = _c1.shape[2:]
        
        _c4 = F.interpolate(_c4, size=target_size, mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=target_size, mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate and Fuse
        fuse = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
        # Final prediction and upsample to original resolution
        out = self.classifier(fuse)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        return out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SwinFloodNet(num_classes=8, in_channels=6).to(device)
    dummy_input = torch.randn(1, 6, 256, 256).to(device)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
