import torch
import cv2
import os
import shutil
import numpy as np

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directories
output_dir = "detection_results_image"
original_frames_dir = os.path.join(output_dir, "original_frames")
detected_frames_dir = os.path.join(output_dir, "detected_frames")
labels_dir = os.path.join(output_dir, "labels")

for folder in [output_dir, original_frames_dir, detected_frames_dir, labels_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Load YOLOv5 model
model_path = r"E:\Stream_Censor\StreamClear\model\openlogo.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(device)
model.conf = 0.25
model.iou = 0.45
model.classes = None
model.eval()

# Path to input video
video_path = r"E:\Stream_Censor\StreamClear\assets\datasets\videoplayback.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_dir, "output_video.mp4")

# Create video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Save original frame
    original_frame_path = os.path.join(original_frames_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(original_frame_path, frame)

    # Convert frame to RGB (OpenCV uses BGR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    with torch.amp.autocast("cuda"):
        results = model(image)

    detections = results.pandas().xyxy[0]

    # Prepare YOLO format bounding box data
    yolo_label_path = os.path.join(labels_dir, f"frame_{frame_count:04d}.txt")
    height, width, _ = frame.shape  # Get frame dimensions

    with open(yolo_label_path, "w") as label_file:
        for _, detection in detections.iterrows():
            xmin, ymin, xmax, ymax = map(int,
                                         [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
            class_id = int(detection['class'])  # Get class number

            # Convert to YOLO format (normalized)
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

            # Draw bounding boxes
            color = (0, 255, 0)  # Green bounding box
            thickness = 2
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)

            # Add label
            label = f"{detection['name']} ({class_id}) {detection['confidence']:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Save detected frame with bounding boxes
    detected_frame_path = os.path.join(detected_frames_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(detected_frame_path, frame)

    # Write frame to output video
    out.write(frame)

cap.release()
out.release()

print(f"\nProcessing complete!")
print(f"Video saved at: {output_video_path}")
print(f"Original frames saved in: {original_frames_dir}")
print(f"Detected frames saved in: {detected_frames_dir}")
print(f"Bounding box labels (YOLO format) saved in: {labels_dir}")
