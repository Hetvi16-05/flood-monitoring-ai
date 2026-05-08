import torch
import torch.nn as nn

class FloodLSTM(nn.Module):
    """
    LSTM-based Flood Risk Predictor.
    Predicts the risk score for the next time step based on a sequence of historical data.
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3, dropout=0.2):
        """
        Args:
            input_size: Number of features per time step (e.g., [water_p, rain_mm, risk_score])
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Number of output features (predicting multiple values: e.g., +1h, +3h, +6h risk)
            dropout: Dropout rate
        """
        super(FloodLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input sequence [Batch, Seq_Len, Input_Size]
        Returns:
            out: Predicted risk score [Batch, Output_Size]
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out

if __name__ == "__main__":
    # Test the model
    model = FloodLSTM(input_size=3, hidden_size=64)
    # Dummy input: (Batch=8, Seq_Len=10, Features=3)
    dummy_input = torch.randn(8, 10, 3)
    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Output Sample: {output[0].item():.4f}")
