import json
import cv2
from pathlib import Path

def draw_boxes(video_path, mapping_path, view, save_path):
    with open(mapping_path, "r") as f:
        mappings = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_key = f"frame_{frame_idx}"
        if frame_key in mappings[view]:
            for pid, bbox in mappings[view][frame_key].items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, pid, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

draw_boxes("videos/broadcast.mp4", "outputs/mappings.json", "broadcast", "outputs/visualized/broadcast_annotated.mp4")
draw_boxes("videos/tacticam.mp4", "outputs/mappings.json", "tacticam", "outputs/visualized/tacticam_annotated.mp4")
