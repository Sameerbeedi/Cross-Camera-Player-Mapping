# Player Detection and Cross-Camera Tracking

A computer vision pipeline for detecting and tracking players across multiple camera views in sports videos.

## Project Structure

```
Player_detection/
├── videos/                  # Input video files
│   ├── broadcast.mp4       # Broadcast camera view
│   └── tacticam.mp4        # Tactical camera view
├── outputs/                 # Generated output files
│   ├── mappings.json       # Player ID mappings
│   └── visualized/         # Visualization outputs
├── data/                   # Detection results
│   ├── broadcast_detections.json
│   └── tacticam_detections.json
├── utils/                  # Utility functions
│   └── video_utils.py      
└── models/                 # ⚠️ Add model files manually
    └── yolov11_player.pt   # YOLOv11 player detection model
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Sameerbeedi/Cross-Camera-Player-Mapping.git
cd Player_detection
```

2. Create models directory:
```bash
mkdir models
```

3. Download the model:
- Download the YOLOv11 player detection model
- Place it in the `models` directory as `yolov11_player.pt`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Add input videos:
- Place your broadcast view video as `videos/broadcast.mp4`
- Place your tactical view video as `videos/tacticam.mp4`

## Running the Pipeline

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Detect players in both videos
2. Extract visual features
3. Match players across camera views
4. Generate visualizations

## Individual Steps

Run steps separately:
```bash
python detectplayers.py    # Player detection
python extract_features.py # Feature extraction
python matching.py        # Cross-camera matching
python visualize.py       # Visualization
```

## Output

- Player detections are saved in `data/`
- Player mappings are saved in `outputs/mappings.json`
- Visualized videos are saved in `outputs/visualized/`

## Dependencies

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv11

## Important Note

⚠️ The YOLOv11 model file must be downloaded and added manually to the `models` directory. The model is not included in this repository due to size constraints.