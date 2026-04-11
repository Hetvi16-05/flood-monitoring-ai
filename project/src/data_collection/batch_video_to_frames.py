import os
import cv2
from pathlib import Path
from datetime import datetime

def batch_extract_frames(input_dir="project/test_videos", output_dir="project/raw_dataset/flood_water", interval_sec=2):
    """
    Iterates through all videos in input_dir and extracts frames every interval_sec.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in input_path.iterdir() if f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"⚠️ No videos found in {input_dir}")
        return

    print(f"🚀 Starting batch extraction for {len(video_files)} videos...")
    print(f"📂 Target: {output_dir}")
    print(f"⏱️ Interval: {interval_sec} seconds")

    total_extracted = 0
    
    for vid_file in video_files:
        print(f"🎬 Processing: {vid_file.name}")
        cap = cv2.VideoCapture(str(vid_file))
        
        if not cap.isOpened():
            print(f"❌ Could not open {vid_file.name}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Fallback
            
        frame_interval = int(fps * interval_sec)
        
        count = 0
        vid_extracted = 0
        video_base_name = vid_file.stem.replace(" ", "_")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                frame_filename = f"batch_{video_base_name}_{vid_extracted:04d}.jpg"
                save_path = output_path / frame_filename
                
                # Check if file exists to avoid duplicate work if re-running
                if not save_path.exists():
                    cv2.imwrite(str(save_path), frame)
                    vid_extracted += 1
                    total_extracted += 1
            
            count += 1
            
        cap.release()
        print(f"✅ Extracted {vid_extracted} frames from {vid_file.name}")

    print(f"\n✨ Batch complete!")
    print(f"📊 Total frames added: {total_extracted}")
    print(f"📁 Local path: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # Default parameters based on user request
    batch_extract_frames()
