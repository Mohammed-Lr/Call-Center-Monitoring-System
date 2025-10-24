import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.detection.drowsiness_detection import DrowsinessDetector
from src.detection.phone_detection import PhoneDetector
from src.detection.headset_detection import HeadsetDetector
from src.detection.face_recognition import FaceRecognitionSystem
import cv2

def demo_drowsiness():
    print("Drowsiness Detection Demo")
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.detect(frame)
        if result:
            frame = detector.annotate_frame(frame, result)
        
        cv2.imshow('Drowsiness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()

def demo_phone_detection(video_path):
    print("Phone Detection Demo")
    detector = PhoneDetector()
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.detect(frame)
        frame = detector.annotate_frame(frame, result)
        
        cv2.imshow('Phone Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def demo_headset_detection(video_path):
    print("Headset Detection Demo")
    detector = HeadsetDetector()
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.detect(frame)
        frame = detector.annotate_frame(frame, result)
        
        cv2.imshow('Headset Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def demo_face_recognition():
    print("Face Recognition Demo")
    recognizer = FaceRecognitionSystem()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = recognizer.recognize_faces(frame)
        frame = recognizer.annotate_frame(frame, faces)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', choices=['drowsiness', 'phone', 'headset', 'face'], required=True)
    parser.add_argument('--video', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.demo == 'drowsiness':
        demo_drowsiness()
    elif args.demo == 'phone':
        demo_phone_detection(args.video)
    elif args.demo == 'headset':
        demo_headset_detection(args.video)
    elif args.demo == 'face':
        demo_face_recognition()