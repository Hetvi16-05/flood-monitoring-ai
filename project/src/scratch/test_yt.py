import os
import sys
from pathlib import Path

# Add project/src to sys.path
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "project" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def test_youtube():
    url = 'https://www.youtube.com/watch?v=S0yYND4j8Qc'
    try:
        import yt_dlp
        save_dir = os.path.join(ROOT, "project", "test_videos")
        os.makedirs(save_dir, exist_ok=True)
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(save_dir, 'test_video.%(ext)s'),
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"🎬 Downloading {url}...")
            info = ydl.extract_info(url, download=True)
            yt_path = ydl.prepare_filename(info)
            print(f"✅ Success! Saved to {yt_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_youtube()
