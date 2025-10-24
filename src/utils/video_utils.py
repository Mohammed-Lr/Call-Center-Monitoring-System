import cv2
from pathlib import Path

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    properties = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    
    cap.release()
    return properties

def save_frame(frame, output_path):
    cv2.imwrite(str(output_path), frame)

def create_video_writer(output_path, fps, width, height, codec='mp4v'):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

def extract_frames(video_path, output_dir, frame_interval=1):
    cap = cv2.VideoCapture(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            output_path = output_dir / f"frame_{saved_count:06d}.jpg"
            save_frame(frame, output_path)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count

def resize_video(input_path, output_path, width, height):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = create_video_writer(output_path, fps, width, height)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (width, height))
        out.write(resized_frame)
    
    cap.release()
    out.release()