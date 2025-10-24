import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.monitoring.pipeline import MonitoringPipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run Call Center Monitoring System')
    parser.add_argument('--video', type=str, default='0', help='Video source (0 for webcam or path to video file)')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--log', type=str, default='output/logs/alerts.log', help='Alert log file path')
    
    args = parser.parse_args()
    
    video_source = 0 if args.video == '0' else args.video
    
    pipeline = MonitoringPipeline()
    
    print("Starting Call Center Monitoring System...")
    print("Press 'q' to quit")
    
    pipeline.process_video(video_source, args.output)
    
    if args.log:
        pipeline.save_alerts_log(args.log)
        print(f"Alerts saved to {args.log}")

if __name__ == "__main__":
    main()