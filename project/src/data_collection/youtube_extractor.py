import os
import requests
from pathlib import Path

PEXELS_API_KEY = "W1SAen94jRy0ZDSjjjoF6UbLhp56bfy9d8sV7NOd2o8G5fQsIFybyeRu"
QUERY = "flood"
PER_PAGE = 100 # number of videos

SAVE_DIR = "project/test_videos"
os.makedirs(SAVE_DIR, exist_ok=True)

headers = {
    "Authorization": PEXELS_API_KEY
}

url = f"https://api.pexels.com/videos/search?query={QUERY}&per_page={PER_PAGE}"

response = requests.get(url, headers=headers)
data = response.json()

print("⬇️ Downloading videos...")

for i, video in enumerate(data["videos"]):
    video_files = video["video_files"]
    
    # pick medium quality
    video_url = video_files[0]["link"]

    video_path = os.path.join(SAVE_DIR, f"pexels_{i}.mp4")

    r = requests.get(video_url, stream=True)
    with open(video_path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)

    print(f"✅ Downloaded: {video_path}")