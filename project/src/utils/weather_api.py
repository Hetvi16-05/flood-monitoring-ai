import requests
import sys
import os
try:
    from config import WEATHER_API_KEY
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    try:
        from config import WEATHER_API_KEY
    except ImportError:
        WEATHER_API_KEY = None
from utils.logger import get_logger
logger = get_logger("WeatherAPI")
def get_rainfall(lat: float, lon: float) -> float:
    """
    Get current rainfall in mm using OpenWeatherMap.
    Must return float. Return 0 if error or API key missing.
    No crash allowed.
    """
    if not WEATHER_API_KEY or WEATHER_API_KEY == "your_openweathermap_api_key":
        return 0.0       
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "rain" in data:
            return float(data["rain"].get("1h", 0.0))
    except Exception as e:
        logger.error(f"Weather API Error: {e}")
    return 0.0