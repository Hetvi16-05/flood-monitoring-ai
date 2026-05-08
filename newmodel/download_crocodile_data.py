import os
from roboflow import Roboflow

def download_crocodile_dataset():
    print("🐊 RAINWISE - CROCODILE DATASET DOWNLOADER")
    print("------------------------------------------")
    print("1. Go to: https://app.roboflow.com/settings/api")
    print("2. Copy your 'Private API Key'")
    
    api_key = input("\n🔑 Paste your Roboflow API Key: ").strip()
    
    if not api_key:
        print("❌ Error: API Key is required.")
        return

    try:
        rf = Roboflow(api_key=api_key)
        # Accessing the TALOV Crocodile Dataset
        project = rf.workspace("talov").project("crocodile-computer-vision-dataset")
        version = project.version(1)
        
        print(f"\n📡 Connecting to Roboflow... Downloading {project.name} v{version.version}")
        
        # Download to a specific folder for our project
        dataset = version.download(
            model_format="yolov8", 
            location="newmodel/crocodile_dataset"
        )
        
        print(f"\n✅ SUCCESS! Dataset downloaded to: {dataset.location}")
        print("You can now use this to fine-tune your Crocodile Detector!")
        
    except Exception as e:
        print(f"\n❌ Download Failed: {e}")

if __name__ == "__main__":
    download_crocodile_dataset()
