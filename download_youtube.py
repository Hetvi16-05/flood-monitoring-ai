import yt_dlp
import os
import sys

def download_youtube_video(url, output_dir="project/test_videos"):
    """
    Downloads a YouTube video to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"🎬 Downloading: {url}")
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', 'video')
            ext = info.get('ext', 'mp4')
            final_path = os.path.join(output_dir, f"{video_title}.{ext}")
            print(f"✅ Success! Saved to: {final_path}")
            return final_path
    except Exception as e:
        print(f"❌ Error downloading video: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_youtube.py <YOUTUBE_URL>")
    else:
        video_url = sys.argv[1]
        download_youtube_video(video_url)
