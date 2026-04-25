import os
import sys
import cv2
import subprocess
from pathlib import Path
from tqdm import tqdm

class YouTubeFloodCollector:
    """
    Automated YouTube Downloader and Frame Extractor for Flood Datasets.
    """
    def __init__(self, output_dir, frame_interval=5):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.videos_tmp = self.output_dir / "videos_tmp"
        self.frame_interval = frame_interval # seconds
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.videos_tmp.mkdir(parents=True, exist_ok=True)

    def download_video(self, url):
        """Downloads a video using yt-dlp."""
        print(f"📥 Downloading: {url}")
        output_template = str(self.videos_tmp / "%(title)s.%(ext)s")
        python_path = sys.executable
        try:
            subprocess.run([
                python_path, "-m", "yt_dlp", 
                "-f", "best",
                "-o", output_template,
                url
            ], check=True)
            return True
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            return False

    def extract_frames(self):
        """Extracts frames from all downloaded videos."""
        videos = list(self.videos_tmp.glob("*.*"))
        print(f"⚙️  Extracting frames from {len(videos)} videos...")
        
        for video_path in tqdm(videos):
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: continue
            
            frame_gap = int(fps * self.frame_interval)
            count = 0
            success = True
            
            video_name = video_path.stem.replace(" ", "_")
            
            while success:
                success, frame = cap.read()
                if not success: break
                
                if count % frame_gap == 0:
                    frame_name = f"{video_name}_f{count}.jpg"
                    save_path = self.images_dir / frame_name
                    cv2.imwrite(str(save_path), frame)
                
                count += 1
            
            cap.release()
            os.remove(video_path)

    def run(self, urls):
        for url in urls:
            if self.download_video(url):
                self.extract_frames()

if __name__ == "__main__":
    FLOOD_VIDEO_URLS = [
        "https://www.youtube.com/watch?v=3-M9X2Mog3g"  # Libya floods (often includes drone footage)
    ]
    
    collector = YouTubeFloodCollector(output_dir="/Users/HetviSheth/Flood_Prediction/project/raw_dataset")
    collector.run(FLOOD_VIDEO_URLS)
