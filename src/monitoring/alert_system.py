import datetime
from ..config import ALERT_PHONE_FRAMES, ALERT_HEADSET_FRAMES, ALERT_DROWSY_FRAMES

class AlertSystem:
    def __init__(self):
        self.phone_frames_count = 0
        self.no_headset_frames_count = 0
        self.drowsy_frames_count = 0
        
        self.phone_alert_active = False
        self.headset_alert_active = False
        self.drowsy_alert_active = False
        
        self.alerts_log = []

    def update(self, phone_detected, headset_detected, is_drowsy, person_name="Unknown"):
        timestamp = datetime.datetime.now()
        
        if phone_detected:
            self.phone_frames_count += 1
        else:
            self.phone_frames_count = 0
            self.phone_alert_active = False

        if not headset_detected:
            self.no_headset_frames_count += 1
        else:
            self.no_headset_frames_count = 0
            self.headset_alert_active = False

        if is_drowsy:
            self.drowsy_frames_count += 1
        else:
            self.drowsy_frames_count = 0
            self.drowsy_alert_active = False

        alerts = []

        if self.phone_frames_count >= ALERT_PHONE_FRAMES and not self.phone_alert_active:
            self.phone_alert_active = True
            alert = {
                'type': 'phone_usage',
                'person': person_name,
                'timestamp': timestamp,
                'message': f'Phone usage detected for {person_name}'
            }
            alerts.append(alert)
            self.alerts_log.append(alert)

        if self.no_headset_frames_count >= ALERT_HEADSET_FRAMES and not self.headset_alert_active:
            self.headset_alert_active = True
            alert = {
                'type': 'no_headset',
                'person': person_name,
                'timestamp': timestamp,
                'message': f'No headset detected for {person_name}'
            }
            alerts.append(alert)
            self.alerts_log.append(alert)

        if self.drowsy_frames_count >= ALERT_DROWSY_FRAMES and not self.drowsy_alert_active:
            self.drowsy_alert_active = True
            alert = {
                'type': 'drowsiness',
                'person': person_name,
                'timestamp': timestamp,
                'message': f'Drowsiness detected for {person_name}'
            }
            alerts.append(alert)
            self.alerts_log.append(alert)

        return alerts

    def get_status(self):
        return {
            'phone_alert': self.phone_alert_active,
            'headset_alert': self.headset_alert_active,
            'drowsy_alert': self.drowsy_alert_active,
            'phone_frames': self.phone_frames_count,
            'no_headset_frames': self.no_headset_frames_count,
            'drowsy_frames': self.drowsy_frames_count
        }

    def reset(self):
        self.phone_frames_count = 0
        self.no_headset_frames_count = 0
        self.drowsy_frames_count = 0
        self.phone_alert_active = False
        self.headset_alert_active = False
        self.drowsy_alert_active = False

    def save_log(self, filepath):
        with open(filepath, 'w') as f:
            for alert in self.alerts_log:
                f.write(f"{alert['timestamp']} - {alert['type']} - {alert['person']} - {alert['message']}\n")