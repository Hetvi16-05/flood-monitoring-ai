import os
import hashlib
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try importing from config, fallback to default if not available
try:
    from config import RAW_DIR
except ImportError:
    RAW_DIR = "project/raw_dataset"

def get_hash(file_path):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def remove_duplicates():
    print(f"🔍 Scanning for duplicates in: {RAW_DIR}")
    
    if not os.path.exists(RAW_DIR):
        print(f"❌ Directory {RAW_DIR} does not exist.")
        return

    hashes = {}
    duplicates = []
    
    # Get all image files
    all_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, file))

    print(f"📊 Total images found: {len(all_files)}")
    
    for file_path in tqdm(all_files, desc="Checking hashes"):
        file_hash = get_hash(file_path)
        
        if file_hash in hashes:
            duplicates.append((file_path, hashes[file_hash]))
        else:
            hashes[file_hash] = file_path

    if not duplicates:
        print("✅ No duplicates found.")
    else:
        print(f"🗑️ Found {len(duplicates)} duplicates. Removing...")
        for dup_path, original_path in duplicates:
            try:
                os.remove(dup_path)
            except Exception as e:
                print(f"❌ Error removing {dup_path}: {e}")

    print(f"✨ Cleanup complete. {len(duplicates)} files removed.")

if __name__ == "__main__":
    remove_duplicates()
