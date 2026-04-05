import os
import logging
import csv
from datetime import datetime
from pathlib import Path

def get_logger(name):
    """
    Returns a robust logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    try:
        project_root = str(Path(__file__).resolve().parents[3])
        log_dir = os.path.join(project_root, "project", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"production_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"File logging failed: {e}")
        
    return logger

def log_to_csv(water_p, obj_summary, risk_level, risk_score, rain, location):
    """
    PRODUCTION LOGGING:
    Records Timestamp, Water %, Object Summary, Rain, Risk Level, Risk Score, Location
    """
    try:
        project_root = str(Path(__file__).resolve().parents[3])
        csv_path = os.path.join(project_root, "project", "logs", "monitor_log.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        is_new = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["Timestamp", "Water_%", "Object_Summary", "Rain_mm", "Risk_Level", "Risk_Score", "Location"])
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{water_p:.2f}",
                obj_summary if obj_summary else "None",
                rain,
                risk_level,
                f"{risk_score}",
                location
            ])
    except Exception as e:
        print(f"CSV Logging failed: {e}")

def log_performance(logger, action, start_time):
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"PERF: {action} took {duration:.4f} seconds")
