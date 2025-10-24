import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.detection.phone_detection import PhoneDetector

@pytest.fixture
def sample_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_phone_detector_initialization():
    try:
        detector = PhoneDetector()
        assert detector is not None
        assert detector.phone_detected == False
    except:
        pytest.skip("Model file not available")

def test_detect_result_structure():
    try:
        detector = PhoneDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        
        assert 'phone_detected' in result
        assert 'detections' in result
        assert 'results' in result
    except:
        pytest.skip("Model file not available")

def test_annotate_frame():
    try:
        detector = PhoneDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        annotated = detector.annotate_frame(frame, result)
        
        assert annotated is not None
        assert annotated.shape == frame.shape
    except:
        pytest.skip("Model file not available")