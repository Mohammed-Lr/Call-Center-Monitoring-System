# Usage Guide

## Quick Start

### Webcam Monitoring

```bash
python scripts/run_full_monitoring.py --video 0
```

### Video File Processing

```bash
python scripts/run_full_monitoring.py --video path/to/video.mp4 --output output/result.mp4
```

## Individual Detection Modules

### Drowsiness Detection

```bash
python scripts/drowsiness_detection.py
```

### Phone Detection

```bash
python scripts/phone_detection_tester.py
```

### Headset Detection

```bash
python scripts/headset_detection_tester.py
```

### Face Recognition

```bash
python scripts/face_calibration_test.py
```

## Using Examples

### Single Detection Demo

```bash
python examples/single_detection_demo.py --demo drowsiness
python examples/single_detection_demo.py --demo phone --video video.mp4
python examples/single_detection_demo.py --demo headset --video video.mp4
python examples/single_detection_demo.py --demo face
```

### Batch Processing

```bash
python examples/batch_processing_demo.py --input data/sample_videos --output output/processed_videos
```

## Python API Usage

### Full Pipeline

```python
from src.monitoring.pipeline import MonitoringPipeline

pipeline = MonitoringPipeline()
pipeline.process_video(video_source=0, output_path="output.mp4")
pipeline.save_alerts_log("alerts.log")
```

### Individual Detectors

```python
from src.detection.drowsiness_detection import DrowsinessDetector
from src.detection.phone_detection import PhoneDetector
from src.detection.headset_detection import HeadsetDetector
from src.detection.face_recognition import FaceRecognitionSystem

drowsiness_detector = DrowsinessDetector()
phone_detector = PhoneDetector()
headset_detector = HeadsetDetector()
face_recognizer = FaceRecognitionSystem()

result = drowsiness_detector.detect(frame)
```

### Alert System

```python
from src.monitoring.alert_system import AlertSystem

alert_system = AlertSystem()
alerts = alert_system.update(phone_detected=True, headset_detected=False, is_drowsy=False)

for alert in alerts:
    print(alert['message'])
```

## Configuration

Edit `src/config.py` to customize:

```python
LIP_DISTANCE_THRESHOLD = 0.05
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 5
ALERT_PHONE_FRAMES = 30
ALERT_HEADSET_FRAMES = 30
ALERT_DROWSY_FRAMES = 5
```

## Output

- Processed videos: `output/processed_videos/`
- Alert logs: `output/logs/`
- Alerts: `output/alerts/`

## Keyboard Controls

- `q`: Quit monitoring