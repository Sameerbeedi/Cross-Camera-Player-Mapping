import cv2
import json
from ultralytics import YOLO
from pathlib import Path

def detect_players(video_path, model_path, save_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    all_detections = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0: 
                    detections.append({
                        "bbox": box.xyxy[0].tolist(),
                        "conf": float(box.conf[0]),
                        "class": int(box.cls[0])
                    })

        all_detections[f"frame_{frame_idx}"] = detections
        frame_idx += 1

    cap.release()

    with open(save_path, "w") as f:
        json.dump(all_detections, f, indent=2)

# Paths
detect_players("videos/broadcast.mp4", "models/best.pt", "data/broadcast_detections.json")
detect_players("videos/tacticam.mp4", "models/best.pt", "data/tacticam_detections.json")
