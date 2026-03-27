import requests
from project.src.utils.logger import get_logger

logger = get_logger("IPLocation")

def get_location():
    """
    Fetch dynamic location (lat, lon, city) based on IP.
    Returns 0, 0, "Unknown" if error.
    No crash allowed.
    """
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "success":
            return float(data["lat"]), float(data["lon"]), str(data["city"])
        else:
            logger.warning("IP-API failed.")
    except Exception as e:
        logger.error(f"Location Error: {e}")
    
    # Return defaults on error
    return 0, 0, "Unknown"
