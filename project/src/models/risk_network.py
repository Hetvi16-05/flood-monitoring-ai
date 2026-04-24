import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridRiskNetwork(nn.Module):
    """
    Hybrid Risk Assessment Network (MTL).
    Fuses vision features with tabular data (weather, topography) to predict:
    1. Risk Score (0-100)
    2. Risk Level (Low, Medium, High, Extreme)
    3. Evacuation Priority (Low, Medium, High)
    """
    def __init__(self, vision_feature_dim=1024, tabular_dim=10):
        super(HybridRiskNetwork, self).__init__()
        
        # 1. Feature Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(vision_feature_dim + tabular_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # 2. Multi-Task Heads
        # Head A: Risk Score Regression (0-100)
        self.risk_score_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # Scale to 0-1 then multiply by 100
        )
        
        # Head B: Risk Level Classification (4 classes)
        self.risk_level_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        # Head C: Evacuation Priority (3 classes)
        self.evac_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, vision_features, weather_topo_data):
        """
        Args:
            vision_features: [B, 1024] pooled features from Seg/Obj/Temporal models
            weather_topo_data: [B, 10] normalized weather/elevation stats
        """
        # Concatenate features
        combined = torch.cat([vision_features, weather_topo_data], dim=1)
        
        # Shared processing
        shared = self.fusion(combined)
        
        # Predict tasks
        risk_score = self.risk_score_head(shared) * 100
        risk_level = self.risk_level_head(shared)
        evac_priority = self.evac_head(shared)
        
        return {
            'risk_score': risk_score,
            'risk_level_logits': risk_level,
            'evacuation_priority_logits': evac_priority
        }

if __name__ == "__main__":
    model = HybridRiskNetwork(vision_feature_dim=1024, tabular_dim=10)
    # Dummy vision features (sum of poolings from other models)
    v_feat = torch.randn(1, 1024)
    # Dummy tabular data (temp, rain_rate, altitude, etc.)
    t_data = torch.randn(1, 10)
    
    output = model(v_feat, t_data)
    print("Hybrid Risk Network Outputs:")
    for k, v in output.items():
        print(f"  {k}: {v.shape}")
        if 'score' in k:
            print(f"    Sample Value: {v.item():.2f}")
