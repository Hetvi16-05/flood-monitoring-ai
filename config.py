import sys
import os
from pathlib import Path

# ---------------------------------------------------------
# BACKWARD COMPATIBILITY WRAPPER
# This file redirects to project/src/config.py
# ---------------------------------------------------------

# Add project/src to path
src_path = os.path.join(os.path.dirname(__file__), "project", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import everything from the new source of truth
try:
    from config import *
except ImportError:
    # Fallback for some environments
    import project.src.config as config
    globals().update({k: v for k, v in config.__dict__.items() if not k.startswith('_')})