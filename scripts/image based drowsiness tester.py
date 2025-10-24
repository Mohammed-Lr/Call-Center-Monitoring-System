import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Thresholds
LIP_DISTANCE_THRESHOLD = 0.05
EAR_THRESHOLD = 0.2


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


def process_image(image_path, output_dir):
    """Process a single image and save the result with drowsiness detection."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    frame_height, frame_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks in pixel coordinates
                landmarks = {i: (int(lm.x * frame_width), int(lm.y * frame_height))
                             for i, lm in enumerate(face_landmarks.landmark)}

                # Calculate lip distance
                lip_distance = calculate_distance(landmarks[LIPS[0]],
                                                  landmarks[LIPS[1]]) / frame_height

                # Calculate EAR for both eyes
                left_ear = calculate_ear(LEFT_EYE, landmarks)
                right_ear = calculate_ear(RIGHT_EYE, landmarks)
                avg_ear = (left_ear + right_ear) / 2.0

                # Determine drowsiness
                is_drowsy = lip_distance > LIP_DISTANCE_THRESHOLD or avg_ear < EAR_THRESHOLD

                # Display measurements
                cv2.putText(image, f"Lip Distance: {lip_distance:.3f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, f"Avg EAR: {avg_ear:.3f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Display status
                status = "DROWSY" if is_drowsy else "NOT DROWSY"
                cv2.putText(image, f"Status: {status}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255) if is_drowsy else (0, 255, 0), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
        else:
            print(f"No face detected in {image_path}")
            cv2.putText(image, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Create output filename
    input_filename = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{input_filename}_analyzed.jpg")

    # Save the analyzed image
    cv2.imwrite(output_path, image)

    return output_path


def process_directory(input_dir, output_dir):
    """Process all images in a directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image formats
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # Process each image in the directory
    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)
            print(f"\nProcessing: {filename}")

            output_path = process_image(input_path, output_dir)
            if output_path:
                processed_count += 1
                print(f"Saved to: {output_path}")

    return processed_count


if __name__ == "__main__":
    # Directory containing input images
    input_directory = r"C:\Users\EliteBook 840 G4\Desktop\projet ia\input drowsiness"  # Replace with your input directory

    # Directory for saving output images
    output_directory = r"C:\Users\EliteBook 840 G4\Desktop\projet ia\output drowsiness"  # Replace with your output directory

    print("Starting drowsiness detection process...")

    # Process all images
    processed_count = process_directory(input_directory, output_directory)

    print(f"\nProcessing complete!")
    print(f"Processed {processed_count} images")
    print(f"Output saved to: {output_directory}")