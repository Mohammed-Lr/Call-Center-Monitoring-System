import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Thresholds
LIP_DISTANCE_THRESHOLD = 0.05  # Adjust based on testing
EAR_THRESHOLD = 0.2  # Adjust based on testing
CONSECUTIVE_FRAMES = 5  # Number of consecutive frames to confirm drowsiness

# Helper functions
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_landmarks, landmarks):
    vertical_1 = calculate_distance(landmarks[eye_landmarks[1]], landmarks[eye_landmarks[5]])
    vertical_2 = calculate_distance(landmarks[eye_landmarks[2]], landmarks[eye_landmarks[4]])
    horizontal = calculate_distance(landmarks[eye_landmarks[0]], landmarks[eye_landmarks[3]])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Lip landmarks
LIPS = [13, 14]

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    drowsy_frames = 0
    is_drowsy = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks in pixel coordinates
                landmarks = {i: (int(lm.x * frame_width), int(lm.y * frame_height)) for i, lm in enumerate(face_landmarks.landmark)}

                # Calculate lip distance
                lip_distance = calculate_distance(landmarks[LIPS[0]], landmarks[LIPS[1]]) / frame_height

                # Calculate EAR for both eyes
                left_ear = calculate_ear(LEFT_EYE, landmarks)
                right_ear = calculate_ear(RIGHT_EYE, landmarks)
                avg_ear = (left_ear + right_ear) / 2.0

                # Drowsiness logic
                if lip_distance > LIP_DISTANCE_THRESHOLD or avg_ear < EAR_THRESHOLD:
                    drowsy_frames += 1
                else:
                    drowsy_frames = 0

                if drowsy_frames >= CONSECUTIVE_FRAMES:
                    is_drowsy = True
                else:
                    is_drowsy = False

                # Display status
                status = "DROWSY" if is_drowsy else "NOT DROWSY"
                cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_drowsy else (0, 255, 0), 2)

                # Draw landmarks (optional)
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
