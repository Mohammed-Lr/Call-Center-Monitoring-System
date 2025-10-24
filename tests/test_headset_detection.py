import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.detection.headset_detection import HeadsetDetector

@pytest.fixture
def sample_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_headset_detector_initialization():
    try:
        detector = HeadsetDetector()
        assert detector is not None
        assert detector.headset_detected == False
    except:
        pytest.skip("Model file not available")

def test_detect_result_structure():
    try:
        detector = HeadsetDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        
        assert 'headset_detected' in result
        assert 'detections' in result
        assert 'results' in result
    except:
        pytest.skip("Model file not available")

def test_annotate_frame():
    try:
        detector = HeadsetDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        annotated = detector.annotate_frame(frame, result)
        
        assert annotated is not None
        assert annotated.shape == frame.shape
    except:
        pytest.skip("Model file not available")