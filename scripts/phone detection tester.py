import cv2
from ultralytics import YOLO

# Load the YOLO model
model_path = r"C:\Users\EliteBook 840 G4\Desktop\projet ia\pdfinal.pt"
model = YOLO(model_path)

# Path to your video file
video_path = r"C:\Users\EliteBook 840 G4\Desktop\projet ia\pd test vid.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define target resolution
target_width = 640
target_height = 480

# Create VideoWriter object to save the processed video
output_path = 'output_pd_video.mp4'
out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      int(cap.get(cv2.CAP_PROP_FPS)),
                      (target_width, target_height))  # Use target resolution for output

print(f"Processing video. Press 'q' to quit.")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached.")
        break

    # Resize frame to target resolution
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # Run the YOLO model on the resized frame
    results = model(resized_frame)

    # Parse the results and draw detections on the frame
    annotated_frame = results[0].plot()

    # Write the frame to output video
    out.write(annotated_frame)

    # Display the frame
    cv2.imshow("Detection", annotated_frame)

    # Update frame count for progress tracking
    frame_count += 1
    if frame_count % 30 == 0:  # Show progress every 30 frames
        print(f"Processed {frame_count} frames")

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output saved to {output_path}")