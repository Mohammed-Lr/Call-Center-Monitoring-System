import cv2
import numpy as np

def draw_bounding_box(frame, bbox, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                 (x1 + label_size[0], y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def add_text_overlay(frame, text, position, font_scale=1, color=(255, 255, 255), 
                     thickness=2, bg_color=None):
    x, y = position
    
    if bg_color:
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       font_scale, thickness)
        cv2.rectangle(frame, (x, y - text_size[1] - 10),
                     (x + text_size[0], y), bg_color, -1)
    
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness)
    
    return frame

def create_status_panel(width, height, status_info):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    y_offset = 30
    for key, value in status_info.items():
        text = f"{key}: {value}"
        cv2.putText(panel, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
    
    return panel

def draw_alert_banner(frame, alert_message, alert_type='warning'):
    height, width = frame.shape[:2]
    
    colors = {
        'warning': (0, 165, 255),
        'danger': (0, 0, 255),
        'success': (0, 255, 0)
    }
    
    color = colors.get(alert_type, (255, 255, 255))
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 60), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    text_size, _ = cv2.getTextSize(alert_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = (width - text_size[0]) // 2
    cv2.putText(frame, alert_message, (text_x, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def concatenate_frames(frames, layout='horizontal'):
    if layout == 'horizontal':
        return np.hstack(frames)
    elif layout == 'vertical':
        return np.vstack(frames)
    elif layout == 'grid':
        rows = []
        cols = int(np.ceil(np.sqrt(len(frames))))
        for i in range(0, len(frames), cols):
            row_frames = frames[i:i+cols]
            rows.append(np.hstack(row_frames))
        return np.vstack(rows)
    
    return frames[0]