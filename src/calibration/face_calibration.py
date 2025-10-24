import cv2
from ..detection.face_recognition import FaceRecognitionSystem

class FaceCalibration:
    def __init__(self, encodings_path=None):
        self.face_system = FaceRecognitionSystem(encodings_path)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        recognized_faces = self.face_system.recognize_faces(image)
        annotated_image = self.face_system.annotate_frame(image, recognized_faces)
        return annotated_image, recognized_faces

    def process_video(self, video_source=0):
        cap = cv2.VideoCapture(video_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            recognized_faces = self.face_system.recognize_faces(frame)
            annotated_frame = self.face_system.annotate_frame(frame, recognized_faces)

            cv2.imshow('Video', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()