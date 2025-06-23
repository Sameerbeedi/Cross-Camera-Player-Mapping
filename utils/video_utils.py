import cv2
import numpy as np
import os

def get_frame(video_path, frame_num):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def extract_frames(video_path):
    """Extract all frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def draw_boxes(frame, boxes, ids):
    """Draw bounding boxes and IDs on a frame."""
    frame_copy = frame.copy()
    for box, id_str in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_copy, str(id_str), (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame_copy

def save_video(frames, output_path, fps=30.0):
    """Save frames as a video file."""
    if not frames:
        print(f"Error: No frames to save")
        return False

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"Error: Could not create video writer for {output_path}")
            return False

        print(f"Saving video to {output_path}")
        print(f"Number of frames: {len(frames)}")
        
        for i, frame in enumerate(frames):
            out.write(frame)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} frames")

        out.release()
        
        if os.path.exists(output_path):
            print(f"Successfully saved video: {output_path}")
            print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            return True
        else:
            print(f"Error: Video file was not created at {output_path}")
            return False

    except Exception as e:
        print(f"Error saving video: {str(e)}")
        return False
