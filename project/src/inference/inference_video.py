import os
import sys
import torch
import cv2
import numpy as np
from torchvision import transforms, models
from pathlib import Path

# -----------------------------
# PATH FIX (RAINWISE_CV_VADODARA context)
# -----------------------------
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import NUM_CLASSES, WEIGHTS_DIR


COLORS = [
    (0,0,0),      # background
    (255,0,0),    # water
    (0,255,0),    # road
    (0,0,255),    # vehicle
    (255,255,0),  # person
    (0,255,255),  # rain
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(device):
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
    model.aux_classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)

    weights_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
    if not os.path.exists(weights_path):
        print(f"⚠️ Weights not found at {weights_path}")
        return None

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


def color_mask(mask):
    h,w = mask.shape
    out = np.zeros((h,w,3),dtype=np.uint8)
    for i,c in enumerate(COLORS):
        if i < len(COLORS):
            out[mask==i] = c
    return out


def run_video(video_path):
    device = get_device()
    model = load_model(device)
    if model is None: return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)["out"]
            pred = torch.argmax(out,1)[0]

        pred = pred.cpu().numpy()
        mask = color_mask(pred)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        overlay = cv2.addWeighted(frame, 0.6, mask, 0.4, 0)

        cv2.imshow("video", overlay)
        if cv2.waitKey(1) == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Point to a video file if available
    video_file = "test.mp4"
    if os.path.exists(video_file):
        run_video(video_file)
    else:
        print(f"⚠️ Video file {video_file} not found. Change path in script.")
