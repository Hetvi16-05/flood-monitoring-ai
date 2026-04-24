import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBlock(channels, channels, 1, 1, 0)
        self.conv2 = ConvBlock(channels, channels, 3, 1, 1)
        self.shortcut = shortcut

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))

class C3Block(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super(C3Block, self).__init__()
        hidden_channels = out_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = ConvBlock(in_channels, hidden_channels, 1, 1, 0)
        self.conv3 = ConvBlock(2 * hidden_channels, out_channels, 1, 1, 0)
        self.m = nn.Sequential(*(Bottleneck(hidden_channels, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))

class EmergencyDetectionHead(nn.Module):
    """
    Custom Detector Head for Flood Scenarios.
    Outputs: Bounding Boxes, Class Probabilities, and Submersion Confidence.
    """
    def __init__(self, in_channels, num_classes):
        super(EmergencyDetectionHead, self).__init__()
        self.num_classes = num_classes
        
        # Branch 1: Bbox regression (x, y, w, h)
        self.bbox_conv = nn.Conv2d(in_channels, 4, kernel_size=1)
        
        # Branch 2: Classification (boats, people, vehicles, etc.)
        self.cls_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        # Branch 3: Submersion Confidence (0-1 regression)
        # This tells us how much of the object is underwater
        self.submersion_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bbox = self.bbox_conv(x)
        cls_probs = self.cls_conv(x)
        submersion = self.sigmoid(self.submersion_conv(x))
        
        return {
            'bbox': bbox,
            'classes': cls_probs,
            'submersion': submersion
        }

class EmergencyDetector(nn.Module):
    """
    Custom Specialized YOLO for Flood-Specific Objects.
    Backbone: CSPDarknet53
    Neck: PANet
    Head: Submersion-Aware Detection Head
    """
    def __init__(self, num_classes=5):
        super(EmergencyDetector, self).__init__()
        # Simplified CSPDarknet Backbone
        self.stem = ConvBlock(3, 32, 3, 1, 1)
        self.stage1 = ConvBlock(32, 64, 3, 2, 1)
        self.c3_1 = C3Block(64, 64, n=1)
        
        self.stage2 = ConvBlock(64, 128, 3, 2, 1)
        self.c3_2 = C3Block(128, 128, n=2)
        
        self.stage3 = ConvBlock(128, 256, 3, 2, 1)
        self.c3_3 = C3Block(256, 256, n=3)
        
        self.stage4 = ConvBlock(256, 512, 3, 2, 1)
        self.c3_4 = C3Block(512, 512, n=3)
        
        # Multi-scale detection head (using the deeper feature map for now)
        self.head = EmergencyDetectionHead(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.c3_1(x)
        x = self.stage2(x)
        x = self.c3_2(x)
        x = self.stage3(x)
        x = self.c3_3(x)
        x = self.stage4(x)
        x = self.c3_4(x)
        
        return self.head(x)

if __name__ == "__main__":
    model = EmergencyDetector(num_classes=5)
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)
    print("Emergency Detector Outputs:")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")
