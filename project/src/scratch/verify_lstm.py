import sys
from pathlib import Path
import os

# Add project src to path
SRC = Path(__file__).parent.parent
sys.path.append(str(SRC))

from models.predictive_forecaster import PredictiveForecaster
from config import WEIGHTS_DIR

def verify():
    forecaster = PredictiveForecaster()
    model_path = WEIGHTS_DIR / "flood_lstm_v1.pth"
    
    print(f"🔍 Testing PredictiveForecaster with LSTM...")
    if not forecaster.load_lstm(str(model_path)):
        print("❌ Failed to load LSTM model")
        return
        
    # Simulate a sequence of data
    # LSTM needs 12 steps by default
    print("📈 Simulating 15 steps of data...")
    for i in range(15):
        # Gradually increasing risk
        water_p = 20 + i * 5
        rain = 5 + i * 2
        risk_score = 30 + i * 4
        
        prediction = forecaster.predict_flood_risk(
            lat=22.3, 
            lon=73.1, 
            current_risk_score=risk_score,
            current_water_p=water_p,
            current_rain=rain
        )
        
        if i >= 11: # Should start showing LSTM score after 12 steps
            print(f"Step {i+1}: Risk={risk_score}, LSTM Predict={prediction.get('lstm_score', 'N/A')}, Level={prediction['level']}")
            
    print("\n✅ Verification complete!")

if __name__ == "__main__":
    verify()
