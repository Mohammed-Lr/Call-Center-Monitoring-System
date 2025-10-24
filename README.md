# Call Center Compliance Monitoring System (CCMS)

An AI-powered real-time monitoring system designed to ensure compliance in call center environments by detecting phone usage, drowsiness, and headset absence.

## Features

- **Face Recognition**: Identifies individual workers using face encodings
- **Phone Detection**: Detects unauthorized phone usage near worker's face
- **Drowsiness Detection**: Monitors eye and lip movements to detect drowsiness
- **Headset Detection**: Ensures workers are wearing headsets
- **Alert System**: Delayed alerts to prevent false alarms
- **Real-time Processing**: Processes video streams in real-time

## Project Structure

```
call-center-monitoring-system/
├── src/                    # Source code
│   ├── detection/          # Detection modules
│   ├── calibration/        # Face calibration
│   ├── monitoring/         # Monitoring pipeline
│   └── utils/              # Utility functions
├── scripts/                # Standalone scripts
├── models/                 # Trained models
├── data/                   # Datasets
├── output/                 # Output files
└── docs/                   # Documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Face Encodings

```bash
python scripts/face_encodings_generator.py
```

### 2. Run Monitoring System

```bash
python scripts/run_full_monitoring.py --video 0
```

For video file:
```bash
python scripts/run_full_monitoring.py --video path/to/video.mp4 --output output/processed_videos/result.mp4
```

## Models

- **Phone Detection**: YOLOv11 nano model trained on 700+ annotated images
- **Headset Detection**: YOLOv11 custom model
- **Face Recognition**: face_recognition library with pickle encodings
- **Drowsiness Detection**: MediaPipe Face Mesh

## Configuration

Edit `src/config.py` to adjust thresholds:

```python
LIP_DISTANCE_THRESHOLD = 0.05
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 5
ALERT_PHONE_FRAMES = 30
ALERT_HEADSET_FRAMES = 30
ALERT_DROWSY_FRAMES = 5
```

## Alert System

Alerts are triggered when violations persist for a configurable duration:
- Phone detection: 30 frames (default)
- Headset absence: 30 frames (default)
- Drowsiness: 5 frames (default)

## License

MIT License

## Contributors

Your Name