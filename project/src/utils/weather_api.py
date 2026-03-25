import requests
from config import WEATHER_API_KEY
from project.src.utils.logger import get_logger

logger = get_logger("WeatherAPI")

def get_rainfall(lat, lon):
    """
    Get current rainfall in mm using OpenWeatherMap.
    """
    if not WEATHER_API_KEY or WEATHER_API_KEY == "your_openweathermap_api_key":
        return 0.0
        
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        # Check for rain in last hour
        if "rain" in data:
            return data["rain"].get("1h", 0.0)
            
    except Exception as e:
        logger.error(f"Weather API Error: {e}")
        
    return 0.0
