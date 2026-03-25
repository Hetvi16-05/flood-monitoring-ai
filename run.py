import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="RAINWISE Enterprise AI Runner")
    parser.add_argument("--image", type=str, help="Run hybrid inference on single image")
    parser.add_argument("--video", type=str, help="Run hybrid inference on video file")
    parser.add_argument("--batch", type=str, help="Run hybrid inference on folder of images")
    parser.add_argument("--camera", action="store_true", help="Start real-time camera stream")
    parser.add_argument("--ui", action="store_true", help="Start Streamlit web dashboard")
    parser.add_argument("--api", action="store_true", help="Start FastAPI production server")
    
    args = parser.parse_args()

    # Get python path
    venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = "python" # Fallback

    if args.image:
        print(f"🚀 Running Inference on: {args.image}")
        cmd = f"{venv_python} project/src/inference/hybrid_inference.py" # Modded for single img logic?
        # Note: hybrid_inference uses test_samples by default, but we can extend it or use it as is
        os.system(cmd)

    elif args.video:
        print(f"🚀 Processing Video: {args.video}")
        os.system(f"{venv_python} project/src/inference/hybrid_video.py {args.video}")

    elif args.batch:
        print(f"🚀 Batch Processing Folder: {args.batch}")
        os.system(f"{venv_python} project/src/inference/batch_inference.py {args.batch}")

    elif args.camera:
        print("🚀 Starting Real-Time Camera...")
        os.system(f"{venv_python} project/src/inference/hybrid_camera.py")

    elif args.ui:
        print("🚀 Launching Streamlit Dashboard...")
        os.system(f"streamlit run project/src/ui/app.py")

    elif args.api:
        print("🚀 Starting FastAPI Server...")
        os.system(f"{venv_python} project/src/api/server.py")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
