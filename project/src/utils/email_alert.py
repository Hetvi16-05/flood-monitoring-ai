try:
    import yagmail
except ImportError:
    yagmail = None

import os
import time
from config import EMAIL_USER, EMAIL_PASS
from utils.logger import get_logger

logger = get_logger("EmailAlert")

LAST_ALERT_TIME = 0
COOLDOWN_SECONDS = 300  # 5 minutes

def send_flood_alert(risk_level, risk_score, water_p, objects, city="Unknown"):
    """
    Debounced Production Alert Module.
    Sends 1 email maximum per 5 minutes using yagmail.
    """
    global LAST_ALERT_TIME
    current_time = time.time()
    
    # 1. Debouncing Check
    if (current_time - LAST_ALERT_TIME) < COOLDOWN_SECONDS:
        return # Block spam
        
    if yagmail is None:
        logger.warning("yagmail not installed. Email alert skipped.")
        return

    if not EMAIL_USER or not EMAIL_PASS or EMAIL_PASS == "your_app_password":
        logger.warning("Email credentials not set. Skipping.")
        return

    try:
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
        subject = f"🔴 CRITICAL FLOOD ALERT | {city.upper()}"
        body = f"RAINWISE Alert:\nRisk Level: {risk_level} ({risk_score}/100).\nWater Coverage: {water_p:.1f}%.\nDetected Objects: {objects}."
        
        yag.send(to=EMAIL_USER, subject=subject, contents=body)
        logger.info(f"📧 Flood alert email sent to {EMAIL_USER}")
        LAST_ALERT_TIME = current_time
        
    except Exception as e:
        logger.error(f"Email Error: {e}")
