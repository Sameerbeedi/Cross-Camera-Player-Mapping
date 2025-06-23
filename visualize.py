import json
import cv2
import os
from utils.video_utils import extract_frames, draw_boxes, save_video


def visualize(video_path, detection_path, output_path, use_mapping=True):
    frames = extract_frames(video_path)
    with open(detection_path, 'r') as f:
        data = json.load(f)

    annotated = []
    for idx, frame in enumerate(frames):
        key = f"frame_{idx}"
        if key not in data:
            annotated.append(frame)
            continue

        if use_mapping:
            boxes = list(data[key].values())
            ids = list(data[key].keys())
        else:
            boxes = [obj["bbox"] for obj in data[key]]
            ids = [str(i) for i in range(len(boxes))]

        annotated_frame = draw_boxes(frame, boxes, ids)
        annotated.append(annotated_frame)

    save_video(annotated, output_path)


if __name__ == "__main__":
    output_dir = "outputs/visualized"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Check if input files exist
    input_files = {
        "broadcast": "videos/broadcast.mp4",  # Changed from data/ to videos/
        "tacticam": "videos/tacticam.mp4",    # Changed from data/ to videos/
        "mappings": "outputs/mappings.json"
    }

    for name, path in input_files.items():
        if not os.path.exists(path):
            print(f"Error: {name} file not found at {path}")
            exit(1)

    # Visualize mapping results
    try:
        with open("outputs/mappings.json") as f:
            mapping = json.load(f)
            print("Successfully loaded mappings.json")
    except Exception as e:
        print(f"Error loading mappings.json: {str(e)}")
        exit(1)

    visualize("videos/broadcast.mp4", "outputs/mappings.json",  # Changed path
             f"{output_dir}/broadcast_annotated.mp4", use_mapping=True)
    visualize("videos/tacticam.mp4", "outputs/mappings.json",   # Changed path
             f"{output_dir}/tacticam_annotated.mp4", use_mapping=True)

    # OPTIONAL: Visualize detections only
    visualize("videos/broadcast.mp4", "data/broadcast_detections.json",   # Changed path
             "outputs/visualized/broadcast_detections_only.mp4", use_mapping=False)
    visualize("videos/tacticam.mp4", "data/tacticam_detections.json",    # Changed path
             "outputs/visualized/tacticam_detections_only.mp4", use_mapping=False)