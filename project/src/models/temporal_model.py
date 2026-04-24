import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TemporalEvolutionModel(nn.Module):
    """
    Temporal Flood Evolution Model.
    Architecture: 3D CNN Backbone + Transformer Encoder + Predictive Head.
    Input: Sequence of 16 frames [B, 3, 16, H, W]
    Output: Predicted expansion heatmap for next 5 steps [B, 1, 5, H, W]
    """
    def __init__(self, sequence_length=16, predict_steps=5):
        super(TemporalEvolutionModel, self).__init__()
        
        # 1. 3D CNN Backbone (Spatiotemporal features)
        # Using R3D_18 as a base for efficient 3D feature extraction
        r3d = models.video.r3d_18(pretrained=True)
        self.backbone = nn.Sequential(*list(r3d.children())[:-2]) # Keep up to the spatial features
        
        # 2. Transformer Layer for Long-Range Temporal Reasoning
        # Flatten spatial dims into tokens: [B, C, T, H', W'] -> [B*H'*W', T, C]
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 3. Evolution Head (Predicting expansion)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=(1, 1, 1)) # Output 1 channel (expansion probability)
        )
        
        self.predict_steps = predict_steps

    def forward(self, x):
        # x shape: [B, 3, T=16, H=224, W=224]
        
        # 1. Extract Spatiotemporal features
        features = self.backbone(x) # [B, 512, T/8, H/32, W/32]
        
        # 2. Transformer Reasoning
        b, c, t, h_f, w_f = features.shape
        # Reshape for transformer: [Batch*Spatial, Time, Channels]
        x_trans = features.permute(0, 3, 4, 2, 1).contiguous().view(b * h_f * w_f, t, c)
        x_trans = self.transformer(x_trans)
        
        # Reshape back to spatial-temporal volume
        x_vol = x_trans.view(b, h_f, w_f, t, c).permute(0, 4, 3, 1, 2) # [B, 512, t, h_f, w_f]
        
        # 3. Predict Evolution
        # For simplicity, we decode the last few time steps into the predicted expansion
        # but in a real scenario, we might use a recursive approach.
        out = self.decoder(x_vol) # [B, 1, T/?, H, W]
        
        # Resize to target prediction steps (5 frames)
        out = F.interpolate(out, size=(self.predict_steps, x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=False)
        
        return torch.sigmoid(out)

if __name__ == "__main__":
    model = TemporalEvolutionModel(sequence_length=16, predict_steps=5)
    dummy_input = torch.randn(1, 3, 16, 224, 224)
    output = model(dummy_input)
    print(f"Temporal Model Input: {dummy_input.shape}")
    print(f"Temporal Model Output (Prediction for next 5 frames): {output.shape}")
