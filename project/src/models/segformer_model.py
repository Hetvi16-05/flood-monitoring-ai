import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W).contiguous()
        return x

class SegFormerFlood(nn.Module):
    def __init__(self, num_classes=2, in_channels=6, embed_dim=256):
        super().__init__()
        
        # 1. PVTv2-B0 Backbone (Lite/Fast Version)
        self.backbone = timm.create_model(
            'pvt_v2_b0', 
            pretrained=True, 
            in_chans=in_channels, 
            features_only=True
        )
        
        # PVTv2-B0 dims: [32, 64, 160, 256]
        dims = [32, 64, 160, 256]
        self.linear_c4 = MLP(input_dim=dims[3], embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=dims[2], embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=dims[1], embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=dims[0], embed_dim=embed_dim)
        
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
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
        
        fuse = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        out = self.classifier(fuse)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
