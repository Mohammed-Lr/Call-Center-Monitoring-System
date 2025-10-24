import cv2
import mediapipe as mp
import numpy as np
from ..config import (
    LIP_DISTANCE_THRESHOLD,
    EAR_THRESHOLD,
    CONSECUTIVE_FRAMES,
    LEFT_EYE,
    RIGHT_EYE,
    LIPS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    MAX_NUM_FACES
)

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        )
        self.drowsy_frames = 0
        self.is_drowsy = False

    def calculate_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def calculate_ear(self, eye_landmarks, landmarks):
        vertical_1 = self.calculate_distance(
            landmarks[eye_landmarks[1]], 
            landmarks[eye_landmarks[5]]
        )
        vertical_2 = self.calculate_distance(
            landmarks[eye_landmarks[2]], 
            landmarks[eye_landmarks[4]]
        )
        horizontal = self.calculate_distance(
            landmarks[eye_landmarks[0]], 
            landmarks[eye_landmarks[3]]
        )
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def detect(self, frame):
        frame_height, frame_width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = {
                    i: (int(lm.x * frame_width), int(lm.y * frame_height)) 
                    for i, lm in enumerate(face_landmarks.landmark)
                }

                lip_distance = self.calculate_distance(
                    landmarks[LIPS[0]], 
                    landmarks[LIPS[1]]
                ) / frame_height

                left_ear = self.calculate_ear(LEFT_EYE, landmarks)
                right_ear = self.calculate_ear(RIGHT_EYE, landmarks)
                avg_ear = (left_ear + right_ear) / 2.0

                if lip_distance > LIP_DISTANCE_THRESHOLD or avg_ear < EAR_THRESHOLD:
                    self.drowsy_frames += 1
                else:
                    self.drowsy_frames = 0

                if self.drowsy_frames >= CONSECUTIVE_FRAMES:
                    self.is_drowsy = True
                else:
                    self.is_drowsy = False

                return {
                    'is_drowsy': self.is_drowsy,
                    'lip_distance': lip_distance,
                    'avg_ear': avg_ear,
                    'drowsy_frames': self.drowsy_frames,
                    'face_landmarks': face_landmarks
                }

        return None

    def annotate_frame(self, frame, detection_result):
        if detection_result:
            status = "DROWSY" if detection_result['is_drowsy'] else "NOT DROWSY"
            color = (0, 0, 255) if detection_result['is_drowsy'] else (0, 255, 0)
            
            cv2.putText(
                frame, 
                f"Status: {status}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                color, 
                2
            )
            
            if detection_result['face_landmarks']:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    detection_result['face_landmarks'],
                    self.mp_face_mesh.FACEMESH_TESSELATION
                )
        
        return frame

    def release(self):
        self.face_mesh.close()