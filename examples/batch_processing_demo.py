import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.monitoring.pipeline import MonitoringPipeline
import os

def process_video_batch(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = MonitoringPipeline()
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(video_extensions)]
    
    print(f"Found {len(video_files)} videos to process")
    
    for i, video_file in enumerate(video_files, 1):
        input_path = os.path.join(input_dir, video_file)
        output_filename = f"processed_{video_file}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nProcessing {i}/{len(video_files)}: {video_file}")
        
        try:
            pipeline.process_video(input_path, output_path)
            
            log_filename = f"alerts_{Path(video_file).stem}.log"
            log_path = os.path.join(output_dir, log_filename)
            pipeline.save_alerts_log(log_path)
            
            print(f"Completed: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
    
    print("\nBatch processing complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process videos')
    parser.add_argument('--input', type=str, required=True, help='Input directory with videos')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    process_video_batch(args.input, args.output)