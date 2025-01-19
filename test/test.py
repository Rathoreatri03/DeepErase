import torch
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

# # Load class names

# with open('openlogo_classes.txt', 'r') as f:
#     class_names = [line.strip() for line in f.readlines()]

# Load model
model_path = "openlogo.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Set model parameters
model.conf = 0.25  # Confidence threshold
model.iou = 0.45   # NMS IOU threshold
model.classes = None  # All classes
model.eval()

# Load and process image
image_path = "1.jpg"
# Read image with OpenCV
original_image = cv2.imread(image_path)
image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image_pil = Image.fromarray(image)

# Perform detection
results = model(image_pil)

# Process results
detections = results.pandas().xyxy[0]  # Get detections as pandas dataframe

# Print results and draw boxes
for idx, detection in detections.iterrows():
    print(f"Detected {detection['name']} with confidence: {detection['confidence']:.2f}")
    print(f"Bounding box: ({detection['xmin']:.0f}, {detection['ymin']:.0f}), ({detection['xmax']:.0f}, {detection['ymax']:.0f})")
    print("-----------------------")
    
    # Draw rectangle on image
    start_point = (int(detection['xmin']), int(detection['ymin']))
    end_point = (int(detection['xmax']), int(detection['ymax']))
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2
    original_image = cv2.rectangle(original_image, start_point, end_point, color, thickness)
    
    # Add label
    label = f"{detection['name']} {detection['confidence']:.2f}"
    cv2.putText(original_image, label, (start_point[0], start_point[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

# Save the annotated image
cv2.imwrite('output.jpg', original_image)

print("Detection complete. Results saved to 'output.jpg'")