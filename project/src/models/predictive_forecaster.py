"""
Predictive Flood Forecasting using Weather API and Time-Series Analysis
Predicts flood risk 1-6 hours ahead using weather data and historical patterns
"""
import requests
import numpy as np
import torch
from datetime import datetime, timedelta
from collections import deque
import os
import sys
from pathlib import Path

# Add project src to path if needed
sys.path.append(str(Path(__file__).parent.parent))
from models.flood_lstm import FloodLSTM


class PredictiveForecaster:
    """
    Predictive flood forecasting using weather API and time-series analysis
    Provides early warning for potential flood events
    """
    
    def __init__(self, api_key=None, history_length=24):
        """
        Initialize predictive forecaster
        
        Args:
            api_key: Weather API key (OpenWeatherMap or similar)
            history_length: Number of historical data points to keep
        """
        self.api_key = api_key
        self.history_length = history_length
        self.risk_history = deque(maxlen=history_length)
        self.weather_history = deque(maxlen=history_length)
        
        # Weather API endpoint (OpenWeatherMap)
        self.weather_url = "https://api.openweathermap.org/data/2.5/forecast"
        
        # LSTM Model
        self.lstm_model = None
        self.lstm_config = {}
        self.lstm_normalization = {}
        self.data_history = deque(maxlen=history_length) # Stores [water_p, rain_mm, risk_score]
    
    def fetch_weather_data(self, lat, lon):
        """
        Fetch weather forecast data from API
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            forecast_data: Weather forecast data
        """
        if not self.api_key or self.api_key == "your_openweathermap_api_key":
            return self._generate_mock_weather_data()
        
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.weather_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except Exception as e:
            print(f"❌ Weather API error: {e}")
            return None
    
    def _parse_weather_data(self, data):
        """
        Parse weather API response
        
        Args:
            data: Raw API response
            
        Returns:
            parsed_data: Parsed weather forecast
        """
        forecast = []
        
        for item in data.get('list', []):
            forecast_item = {
                'timestamp': datetime.fromtimestamp(item['dt']),
                'rainfall': item.get('rain', {}).get('3h', 0),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'wind_speed': item['wind']['speed'],
                'pressure': item['main']['pressure']
            }
            forecast.append(forecast_item)
        
        return forecast

    def _generate_mock_weather_data(self):
        """
        Generate realistic mock weather data for demonstration purposes
        """
        mock_data = []
        base_time = datetime.now()
        
        # Current conditions
        mock_data.append({
            'timestamp': base_time,
            'rainfall': 15.5,
            'temperature': 28.5,
            'humidity': 88,
            'wind_speed': 12.5,
            'pressure': 1005
        })
        
        # Forecast for next 24 hours (every 3 hours)
        for i in range(1, 8):
            time_offset = i * 3
            # Increase rainfall for next 6-9 hours
            rainfall = 25.0 if 3 <= time_offset <= 9 else 5.0
            
            mock_data.append({
                'timestamp': base_time + timedelta(hours=time_offset),
                'rainfall': rainfall,
                'temperature': 26.0 - (i * 0.5),
                'humidity': min(95, 88 + (i * 1)),
                'wind_speed': 15.0 + i,
                'pressure': 1005 - i
            })
            
        print("💡 Using MOCK weather data (No API Key provided)")
        return mock_data
    
    def calculate_flood_probability(self, weather_data, current_risk_score):
        """
        Calculate flood probability based on weather forecast
        
        Args:
            weather_data: Weather forecast data
            current_risk_score: Current flood risk score
            
        Returns:
            probability: Flood probability (0-1)
            risk_factors: List of contributing factors
        """
        if not weather_data:
            return 0.0, ["No weather data available"]
        
        risk_factors = []
        probability = 0.0
        
        # Analyze next 6 hours
        next_6h = [w for w in weather_data if w['timestamp'] <= datetime.now() + timedelta(hours=6)]
        
        # Total rainfall in next 6 hours
        total_rainfall = sum(w['rainfall'] for w in next_6h)
        
        if total_rainfall > 50:
            probability += 0.4
            risk_factors.append(f"Heavy rain expected: {total_rainfall}mm in 6h")
        elif total_rainfall > 25:
            probability += 0.2
            risk_factors.append(f"Moderate rain expected: {total_rainfall}mm in 6h")
        
        # High humidity
        avg_humidity = np.mean([w['humidity'] for w in next_6h])
        if avg_humidity > 85:
            probability += 0.15
            risk_factors.append(f"High humidity: {avg_humidity:.0f}%")
        
        # Low pressure (indicates storm systems)
        avg_pressure = np.mean([w['pressure'] for w in next_6h])
        if avg_pressure < 1000:
            probability += 0.1
            risk_factors.append(f"Low pressure: {avg_pressure:.0f}hPa")
        
        # Current risk influence
        if current_risk_score > 70:
            probability += 0.2
            risk_factors.append("Current high flood risk")
        elif current_risk_score > 40:
            probability += 0.1
            risk_factors.append("Current moderate flood risk")
        
        # Cap probability at 1.0
        probability = min(1.0, probability)
        
        return probability, risk_factors
    
    def load_lstm(self, model_path):
        """
        Load trained LSTM model and its configuration
        """
        if not os.path.exists(model_path):
            print(f"⚠️ LSTM Model not found at {model_path}")
            return False
            
        try:
            # Note: weights_only=False is used because the checkpoint contains numpy arrays for normalization
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            config = checkpoint['config']
            
            self.lstm_model = FloodLSTM(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=config.get('output_size', 1)
            )
            self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
            self.lstm_model.eval()
            
            self.lstm_config = config
            self.lstm_normalization = {
                'min': checkpoint['min_val'],
                'max': checkpoint['max_val']
            }
            print(f"✅ LSTM Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
            return False

    def predict_flood_risk(self, lat, lon, current_risk_score, current_water_p=0, current_rain=0):
        """
        Predict flood risk for next 1-6 hours
        
        Args:
            lat: Latitude
            lon: Longitude
            current_risk_score: Current flood risk score
            current_water_p: Current water percentage (from vision)
            current_rain: Current rainfall mm
            
        Returns:
            prediction: Dictionary with prediction results
        """
        # Store in history for LSTM
        self.data_history.append([current_water_p, current_rain, current_risk_score])
        
        # Fetch weather data
        weather_data = self.fetch_weather_data(lat, lon)
        
        # Calculate rule-based probability (baseline)
        probability, risk_factors = self.calculate_flood_probability(weather_data, current_risk_score)
        
        # Override with LSTM if enough history is available
        lstm_forecast = None
        if self.lstm_model and len(self.data_history) >= self.lstm_config.get('seq_length', 12):
            lstm_forecast = self._predict_with_lstm()
            if lstm_forecast is not None:
                # Use the immediate next step (+1h) for current probability adjustment
                next_risk = lstm_forecast[0]
                probability = (probability + (next_risk / 100.0)) / 2
                risk_factors.append(f"LSTM Forecast (+1h): {next_risk:.1f} predicted risk")
                if len(lstm_forecast) > 1:
                    risk_factors.append(f"LSTM Forecast (+6h): {lstm_forecast[-1]:.1f} predicted risk")

        # Store in history
        self.risk_history.append(current_risk_score)
        if weather_data:
            self.weather_history.append(weather_data)
        
        # Determine prediction level
        if probability > 0.7:
            level = "DANGEROUS"
            hours_ahead = 2
        elif probability > 0.5:
            level = "HIGH"
            hours_ahead = 4
        elif probability > 0.3:
            level = "MEDIUM"
            hours_ahead = 6
        else:
            level = "LOW"
            hours_ahead = 6
        
        prediction = {
            'probability': probability,
            'level': level,
            'hours_ahead': hours_ahead,
            'risk_factors': risk_factors,
            'timestamp': datetime.now(),
            'lstm_score': lstm_forecast[0] if lstm_forecast else None,
            'forecast_sequence': lstm_forecast,
            'emergency': self.check_for_emergencies(lstm_forecast) if lstm_forecast else None
        }
        
        return prediction

    def _predict_with_lstm(self):
        """
        Internal method to run LSTM inference
        """
        try:
            seq_len = self.lstm_config['seq_length']
            data = np.array(list(self.data_history))[-seq_len:]
            
            # Normalize
            data = (data - self.lstm_normalization['min']) / (self.lstm_normalization['max'] - self.lstm_normalization['min'] + 1e-6)
            
            # Convert to tensor
            input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0) # [1, seq, features]
            
            with torch.no_grad():
                output = self.lstm_model(input_tensor) # [1, output_size]
                
            # Denormalize (risk score is index 2)
            # Normalization was (val - min) / (max - min)
            # So val = norm * (max - min) + min
            pred_norms = output.squeeze().cpu().numpy()
            if pred_norms.ndim == 0:
                pred_norms = np.array([pred_norms])
                
            min_risk = self.lstm_normalization['min'][2]
            max_risk = self.lstm_normalization['max'][2]
            pred_scores = pred_norms * (max_risk - min_risk) + min_risk
            
            return [float(s) for s in pred_scores]
        except Exception as e:
            print(f"⚠️ LSTM Prediction failed: {e}")
            return None
    
    def get_early_warning(self, prediction):
        """
        Generate early warning message based on prediction
        
        Args:
            prediction: Prediction result
            
        Returns:
            warning: Warning message or None
        """
        if prediction['level'] in ['HIGH', 'DANGEROUS']:
            warning = f"⚠️ FLOOD WARNING: {prediction['level']} risk predicted in {prediction['hours_ahead']} hours"
            if prediction['risk_factors']:
                warning += "\nFactors: " + ", ".join(prediction['risk_factors'])
            return warning
        elif prediction['level'] == 'MEDIUM':
            return f"⚡ FLOOD WATCH: Moderate risk possible in {prediction['hours_ahead']} hours"
        else:
            return None
    
    def get_risk_trend(self):
        """
        Analyze risk trend from historical data
        
        Returns:
            trend: 'increasing', 'decreasing', 'stable'
            trend_rate: Rate of change
        """
        if len(self.risk_history) < 5:
            return 'stable', 0.0
        
        recent = list(self.risk_history)[-5:]
        if len(recent) < 2:
            return 'stable', 0.0
        
        # Calculate trend
        trend_rate = (recent[-1] - recent[0]) / len(recent)
        
        if trend_rate > 5:
            return 'increasing', trend_rate
        elif trend_rate < -5:
            return 'decreasing', trend_rate
        else:
            return 'stable', trend_rate

    def check_for_emergencies(self, forecast_sequence):
        """
        Check for rapid risk escalation in the forecast
        """
        if not forecast_sequence or len(forecast_sequence) < 2:
            return None
            
        # If risk jumps by more than 30 points between steps, or reaches > 80
        start_risk = forecast_sequence[0]
        end_risk = forecast_sequence[-1]
        
        if end_risk > 85:
            return "CRITICAL: Flood imminent within 6 hours!"
        if end_risk - start_risk > 25:
            return "ALARM: Rapid risk escalation detected!"
            
        return None


def get_default_weather_api_key():
    """
    Get default weather API key from environment or config
    Returns None if not configured
    """
    import os
    return os.getenv('OPENWEATHERMAP_API_KEY')


if __name__ == "__main__":
    print("🔮 Predictive Flood Forecaster Module Ready")
    print("To use: forecaster = PredictiveForecaster(api_key='your_api_key')")
    print("Get free API key from: https://openweathermap.org/api")
