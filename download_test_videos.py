import os
import urllib.request

VIDEOS = {
    "flood1.mp4":
    "https://samplelib.com/lib/preview/mp4/sample-5s.mp4",

    "flood2.mp4":
    "https://samplelib.com/lib/preview/mp4/sample-10s.mp4",

    "flood3.mp4":
    "https://filesamples.com/samples/video/mp4/sample_640x360.mp4",
}

out_dir = "project/test_videos"
os.makedirs(out_dir, exist_ok=True)

for name, url in VIDEOS.items():
    path = os.path.join(out_dir, name)
    print("Downloading", name)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        print(f"Failed to download {name}: {e}")

print("Done")
