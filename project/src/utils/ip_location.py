import requests
from project.src.utils.logger import get_logger

logger = get_logger("IPLocation")

def get_location():
    """
    Fetch dynamic location (lat, lon, city) based on IP.
    """
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()
        if data["status"] == "success":
            return data["lat"], data["lon"], data["city"]
        else:
            logger.warning("IP-API failed, using fallback.")
    except Exception as e:
        logger.error(f"Location Error: {e}")
    
    # Fallback to config defaults
    return 22.3072, 73.1812, "Vadodara"
