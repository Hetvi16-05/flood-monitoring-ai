"""
Predictive Flood Forecasting using Weather API and Time-Series Analysis
Predicts flood risk 1-6 hours ahead using weather data and historical patterns
"""
import requests
import numpy as np
from datetime import datetime, timedelta
from collections import deque


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
    
    def fetch_weather_data(self, lat, lon):
        """
        Fetch weather forecast data from API
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            forecast_data: Weather forecast data
        """
        if not self.api_key:
            return None
        
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
    
    def predict_flood_risk(self, lat, lon, current_risk_score):
        """
        Predict flood risk for next 1-6 hours
        
        Args:
            lat: Latitude
            lon: Longitude
            current_risk_score: Current flood risk score
            
        Returns:
            prediction: Dictionary with prediction results
        """
        # Fetch weather data
        weather_data = self.fetch_weather_data(lat, lon)
        
        # Calculate flood probability
        probability, risk_factors = self.calculate_flood_probability(weather_data, current_risk_score)
        
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
            'timestamp': datetime.now()
        }
        
        return prediction
    
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
