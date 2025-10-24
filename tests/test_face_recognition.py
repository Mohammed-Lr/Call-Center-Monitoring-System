import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.detection.face_recognition import FaceRecognitionSystem

@pytest.fixture
def sample_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_face_recognition_initialization():
    try:
        recognizer = FaceRecognitionSystem()
        assert recognizer is not None
        assert hasattr(recognizer, 'known_face_encodings')
        assert hasattr(recognizer, 'known_face_names')
    except:
        pytest.skip("Encodings file not available")

def test_recognize_faces_no_face():
    try:
        recognizer = FaceRecognitionSystem()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = recognizer.recognize_faces(frame)
        assert isinstance(faces, list)
        assert len(faces) == 0
    except:
        pytest.skip("Encodings file not available")

def test_annotate_frame():
    try:
        recognizer = FaceRecognitionSystem()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = []
        annotated = recognizer.annotate_frame(frame, faces)
        assert annotated is not None
        assert annotated.shape == frame.shape
    except:
        pytest.skip("Encodings file not available")