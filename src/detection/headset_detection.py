import cv2
from ultralytics import YOLO
from ..config import HEADSET_MODEL_PATH

class HeadsetDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = str(HEADSET_MODEL_PATH)
        self.model = YOLO(model_path)
        self.headset_detected = False

    def detect(self, frame):
        results = self.model(frame)
        detections = results[0].boxes
        
        self.headset_detected = len(detections) > 0
        
        return {
            'headset_detected': self.headset_detected,
            'detections': detections,
            'results': results[0]
        }

    def annotate_frame(self, frame, detection_result):
        if detection_result and detection_result['results']:
            annotated_frame = detection_result['results'].plot()
            return annotated_frame
        return frame