import face_recognition
import cv2
import pickle
import numpy as np
from ..config import FACE_ENCODINGS_PATH

class FaceRecognitionSystem:
    def __init__(self, encodings_path=None):
        if encodings_path is None:
            encodings_path = str(FACE_ENCODINGS_PATH)
        
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
            self.known_face_encodings = data["encodings"]
            self.known_face_names = data["names"]

    def recognize_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding
            )
            name = "Unknown"
            
            if True in matches:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            recognized_faces.append({
                'name': name,
                'location': (top, right, bottom, left),
                'encoding': face_encoding
            })
        
        return recognized_faces

    def annotate_frame(self, frame, recognized_faces):
        for face in recognized_faces:
            top, right, bottom, left = face['location']
            name = face['name']
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                name, 
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.75, 
                (0, 255, 0), 
                2
            )
        
        return frame