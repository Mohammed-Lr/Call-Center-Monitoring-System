import cv2
from ..detection.drowsiness_detection import DrowsinessDetector
from ..detection.phone_detection import PhoneDetector
from ..detection.headset_detection import HeadsetDetector
from ..detection.face_recognition import FaceRecognitionSystem
from .alert_system import AlertSystem
from ..config import TARGET_WIDTH, TARGET_HEIGHT

class MonitoringPipeline:
    def __init__(self, phone_model_path=None, headset_model_path=None, face_encodings_path=None):
        self.drowsiness_detector = DrowsinessDetector()
        self.phone_detector = PhoneDetector(phone_model_path)
        self.headset_detector = HeadsetDetector(headset_model_path)
        self.face_recognizer = FaceRecognitionSystem(face_encodings_path)
        self.alert_system = AlertSystem()

    def process_frame(self, frame):
        resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        recognized_faces = self.face_recognizer.recognize_faces(resized_frame)
        person_name = recognized_faces[0]['name'] if recognized_faces else "Unknown"
        
        drowsiness_result = self.drowsiness_detector.detect(resized_frame)
        phone_result = self.phone_detector.detect(resized_frame)
        headset_result = self.headset_detector.detect(resized_frame)
        
        is_drowsy = drowsiness_result['is_drowsy'] if drowsiness_result else False
        phone_detected = phone_result['phone_detected'] if phone_result else False
        headset_detected = headset_result['headset_detected'] if headset_result else False
        
        alerts = self.alert_system.update(phone_detected, headset_detected, is_drowsy, person_name)
        
        annotated_frame = resized_frame.copy()
        annotated_frame = self.face_recognizer.annotate_frame(annotated_frame, recognized_faces)
        
        if drowsiness_result:
            annotated_frame = self.drowsiness_detector.annotate_frame(annotated_frame, drowsiness_result)
        
        status = self.alert_system.get_status()
        y_offset = 120
        
        if status['phone_alert']:
            cv2.putText(annotated_frame, "ALERT: Phone Usage!", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        if status['headset_alert']:
            cv2.putText(annotated_frame, "ALERT: No Headset!", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        if status['drowsy_alert']:
            cv2.putText(annotated_frame, "ALERT: Drowsiness!", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return {
            'frame': annotated_frame,
            'alerts': alerts,
            'person_name': person_name,
            'is_drowsy': is_drowsy,
            'phone_detected': phone_detected,
            'headset_detected': headset_detected,
            'status': status
        }

    def process_video(self, video_source=0, output_path=None):
        cap = cv2.VideoCapture(video_source)
        
        out = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (TARGET_WIDTH, TARGET_HEIGHT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame)
            
            if out:
                out.write(result['frame'])
            
            cv2.imshow('Monitoring', result['frame'])
            
            if result['alerts']:
                for alert in result['alerts']:
                    print(f"[{alert['timestamp']}] {alert['message']}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        self.drowsiness_detector.release()

    def save_alerts_log(self, filepath):
        self.alert_system.save_log(filepath)