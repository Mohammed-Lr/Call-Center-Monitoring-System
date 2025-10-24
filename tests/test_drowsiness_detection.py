import pytest
import numpy as np
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.detection.drowsiness_detection import DrowsinessDetector

@pytest.fixture
def detector():
    return DrowsinessDetector()

@pytest.fixture
def sample_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_detector_initialization(detector):
    assert detector is not None
    assert detector.drowsy_frames == 0
    assert detector.is_drowsy == False

def test_calculate_distance(detector):
    p1 = (0, 0)
    p2 = (3, 4)
    distance = detector.calculate_distance(p1, p2)
    assert distance == 5.0

def test_detect_no_face(detector, sample_frame):
    result = detector.detect(sample_frame)
    assert result is None

def test_annotate_frame(detector, sample_frame):
    annotated = detector.annotate_frame(sample_frame, None)
    assert annotated is not None
    assert annotated.shape == sample_frame.shape

def test_detector_release(detector):
    detector.release()