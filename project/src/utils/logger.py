import os
import logging
from datetime import datetime
from pathlib import Path

# Fix path to config if relative imports fail
project_root = str(Path(__file__).resolve().parents[3])
log_dir = os.path.join(project_root, "project", "logs")
os.makedirs(log_dir, exist_ok=True)

# Generate log filename with timestamp
log_file = os.path.join(log_dir, f"production_{datetime.now().strftime('%Y%m%d')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    """Returns a logger instance for the given module name."""
    return logging.getLogger(name)

# Helper to log timing
def log_performance(logger, action, start_time):
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"PERF: {action} took {duration:.4f} seconds")
