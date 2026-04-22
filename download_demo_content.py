"""
Download demo videos and images for RAINWISE Flood Monitoring AI demo
"""
import yt_dlp
from pathlib import Path
import os

# Recommended flood demo videos from YouTube
DEMO_VIDEOS = [
    {
        "name": "Urban Flooding",
        "url": "https://www.youtube.com/watch?v=example1",  # Replace with actual URLs
        "description": "Flooded streets with vehicles"
    },
    {
        "name": "Flash Flood",
        "url": "https://www.youtube.com/watch?v=example2",
        "description": "Rapid water rise scenario"
    },
    {
        "name": "People in Flood",
        "url": "https://www.youtube.com/watch?v=example3",
        "description": "People wading through flood water"
    }
]

def download_youtube_video(url, output_dir):
    """
    Download video from YouTube
    """
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            print(f"✅ Downloaded: {info['title']}")
            return filename
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return None

def download_demo_videos():
    """
    Download recommended demo videos
    """
    output_dir = Path(__file__).parent / "project" / "test_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading demo videos for RAINWISE Flood AI...")
    print("=" * 60)
    
    # Check if there are already demo videos
    existing_videos = list(output_dir.glob("*.mp4"))
    if existing_videos:
        print(f"✅ Found {len(existing_videos)} existing videos:")
        for video in existing_videos[:5]:
            print(f"   - {video.name}")
        print()
    
    print("📋 RECOMMENDED DEMO VIDEO SOURCES:")
    print("=" * 60)
    print("\n🔍 Search these terms on YouTube:")
    print("   • 'flood footage'")
    print("   • 'flash flood'")
    print("   • 'urban flooding'")
    print("   • 'flood rescue'")
    print("   • 'crocodile in flood'")
    print("\n🎬 FREE STOCK VIDEO SITES:")
    print("   • Pexels: https://www.pexels.com/videos/search/flood/")
    print("   • Pixabay: https://pixabay.com/videos/search/flood/")
    print("   • Videvo: https://www.videvo.net/search?q=flood")
    print("\n📥 TO DOWNLOAD FROM YOUTUBE:")
    print("   python download_youtube.py '<youtube_url>'")
    print()
    
    print("✅ Demo content directory:", output_dir)
    print("\n💡 TIP: You already have these videos:")
    print("   • crocodile_test.mp4 (crocodile in flood)")
    print("   • Multiple pexels_*.mp4 files")
    
    print("\n📊 RECOMMENDED DEMO SCENARIOS:")
    print("   1. LOW RISK: Dry streets, normal conditions")
    print("   2. MEDIUM RISK: Standing water, flooded roads")
    print("   3. HIGH RISK: People/vehicles in water")
    print("   4. DANGEROUS: Crocodiles in flood water")

if __name__ == "__main__":
    download_demo_videos()
