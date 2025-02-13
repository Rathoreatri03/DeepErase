import torch
import cv2
import numpy as np

# Load YOLOv5 Model
model_path = r"E:\Stream_Censor\DeepErase\model\openlogo.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)

# Set Model Parameters
model.conf = 0.60  # Confidence threshold
model.iou = 0.45   # NMS IOU threshold
model.classes = None  # Detect all classes
model.eval()

# Input and Output Video Paths
video_path = r"E:\Stream_Censor\DeepErase\assets\datasets\puma.mp4"
output_path = r"/DeepErase/results\output_video.mp4"
detections_path = r"/DeepErase/results\detections.txt"

# Open Video File
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Could not open input video.")
    exit()

# Get Video Properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define Video Writer (use 'mp4v' codec for better compatibility)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternative: 'MJPG'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Open a single text file for storing detections
with open(detections_path, "w") as file:
    file.write("Frame,Class,Confidence,Xmin,Ymin,Xmax,Ymax\n")  # CSV Header

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("✅ Video processing complete.")
            break

        frame_count += 1
        print(f"📽 Processing frame {frame_count}...")

        # Convert BGR to RGB for YOLOv5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform Object Detection
        results = model(rgb_frame)
        detections = results.pandas().xyxy[0]  # Get detections for the frame

        # Draw Bounding Boxes and Save Detections
        for _, detection in detections.iterrows():
            class_name = detection["name"]
            confidence = detection["confidence"]
            xmin, ymin, xmax, ymax = int(detection["xmin"]), int(detection["ymin"]), int(detection["xmax"]), int(detection["ymax"])

            # Write to detections.txt
            file.write(f"{frame_count},{class_name},{confidence:.2f},{xmin},{ymin},{xmax},{ymax}\n")

            # Draw Rectangle and Label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write Processed Frame to Output Video
        out.write(frame)

cap.release()
out.release()

# Comment or remove the next line if not using GUI
# cv2.destroyAllWindows()

print(f"✅ Detection complete. Processed video saved as: {output_path}")
print(f"✅ Detections saved in: {detections_path}")
