import yagmail
import os
from config import EMAIL_USER, EMAIL_PASS
from project.src.utils.logger import get_logger

logger = get_logger("EmailAlert")

def send_flood_alert(message, city="Unknown"):
    """
    Send automated email alert using yagmail.
    """
    if not EMAIL_USER or not EMAIL_PASS or EMAIL_PASS == "your_app_password":
        logger.warning("Email credentials not set. Skipping.")
        return

    try:
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
        subject = f"CRITICAL FLOOD ALERT | {city.upper()}"
        body = f"RAINWISE System has detected a critical flood level.\n\nDetails: {message}\n\nLocation: {city}"
        
        yag.send(to=EMAIL_USER, subject=subject, contents=body)
        logger.info(f"📧 Flood alert email sent to {EMAIL_USER}")
        
    except Exception as e:
        logger.error(f"Email Error: {e}")
