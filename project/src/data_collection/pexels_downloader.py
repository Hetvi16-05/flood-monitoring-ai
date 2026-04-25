import os
import requests
from pathlib import Path
from tqdm import tqdm

class PexelsFloodCollector:
    """
    Downloads high-quality flood videos from Pexels API.
    """
    def __init__(self, api_key, output_dir):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.videos_tmp = self.output_dir / "videos_tmp"
        self.headers = {"Authorization": self.api_key}
        
        self.videos_tmp.mkdir(parents=True, exist_ok=True)

    def search_videos(self, query, per_page=5):
        print(f"🔍 Searching Pexels for: {query}")
        url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json().get('videos', [])
        else:
            print(f"❌ Pexels Search Failed: {response.status_code}")
            return []

    def download_videos(self, videos):
        for v in tqdm(videos, desc="Pexels Downloads"):
            # Get the best quality (usually first in video_files)
            video_url = v['video_files'][0]['link']
            video_name = f"pexels_{v['id']}.mp4"
            save_path = self.videos_tmp / video_name
            
            print(f"📥 Downloading: {video_name}")
            resp = requests.get(video_url, stream=True)
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk: f.write(chunk)

if __name__ == "__main__":
    # Placeholder API Key - User must provide a valid one
    PEXELS_API_KEY = "YOUR_PEXELS_API_KEY" 
    
    collector = PexelsFloodCollector(api_key=PEXELS_API_KEY, output_dir="/Users/HetviSheth/Flood_Prediction/project/raw_dataset")
    
    # Try searching for flood footage
    videos = collector.search_videos("flood water", per_page=2)
    if videos:
        collector.download_videos(videos)
    else:
        print("💡 Tip: Get a free API key at pexels.com/api to activate stock video scraping.")
