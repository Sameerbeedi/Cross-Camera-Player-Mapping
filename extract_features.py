import json
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import os
import numpy as np

# Create required directories
os.makedirs("data", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# Update model loading to use newer syntax
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    img = transform(Image.fromarray(crop)).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img).squeeze()
    return features.numpy().tolist()

def ensure_file_exists(file_path):
    """Check if file exists and create empty JSON if needed"""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f)

def process_video(video_path, detections_path, save_path):
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Ensure detection file exists
    ensure_file_exists(detections_path)
    
    cap = cv2.VideoCapture(video_path)
    try:
        with open(detections_path, "r") as f:
            detections = json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading {detections_path}. Creating empty detection set.")
        detections = {}

    frame_idx = 0
    all_features = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_key = f"frame_{frame_idx}"
        if frame_key in detections:
            features = []
            for det in detections[frame_key]:
                feat = extract_features(frame, det["bbox"])
                features.append({"bbox": det["bbox"], "feature": feat})
            all_features[frame_key] = features

        frame_idx += 1

    cap.release()

    # Save features
    with open(save_path, "w") as f:
        json.dump(all_features, f, indent=2)
    print(f"Saved features to {save_path}")

if __name__ == "__main__":
    try:
        process_video(
            "videos/broadcast.mp4", 
            "data/broadcast_detections.json", 
            "data/broadcast_features.json"
        )
        process_video(
            "videos/tacticam.mp4", 
            "data/tacticam_detections.json", 
            "data/tacticam_features.json"
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
