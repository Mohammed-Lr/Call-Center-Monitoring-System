import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.monitoring.pipeline import MonitoringPipeline

def main():
    pipeline = MonitoringPipeline()
    
    print("Starting monitoring with webcam...")
    print("Press 'q' to quit")
    
    pipeline.process_video(video_source=0)
    
    pipeline.save_alerts_log("output/logs/quick_start_alerts.log")
    print("Monitoring stopped. Alerts saved.")

if __name__ == "__main__":
    main()