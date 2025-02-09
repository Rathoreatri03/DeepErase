import torch
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import random
import os
import shutil

# Create output directory if it doesn't exist
output_dir = "detection_results"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Remove if exists
os.makedirs(output_dir)

# Load model
model_path = r"E:\Stream_Censor\StreamClear\model\openlogo.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Set model parameters
model.conf = 0.25
model.iou = 0.45
model.classes = None
model.eval()

# Directory containing input images
input_dir = "C:\\Users\\GARV\\OneDrive\\Documents\\project\\logoRemover\\Logo_model\\openlogo\\JPEGImages"  # Change this to your input directory path

# Get list of all image files
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Select 100 random images (or all if less than 100)
num_images = min(100, len(image_files))
selected_images = random.sample(image_files, num_images)

# Process each selected image
for i, image_file in enumerate(selected_images, 1):
    try:
        print(f"Processing image {i}/{num_images}: {image_file}")
        
        # Load and process image
        image_path = os.path.join(input_dir, image_file)
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Failed to load image: {image_file}")
            continue
            
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)

        # Perform detection
        results = model(image_pil)
        detections = results.pandas().xyxy[0]

        # Create text file for detections
        txt_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_detections.txt")
        with open(txt_path, 'w') as f:
            for idx, detection in detections.iterrows():
                # Write detection to file
                f.write(f"Detected {detection['name']} with confidence: {detection['confidence']:.2f}\n")
                f.write(f"Bounding box: ({detection['xmin']:.0f}, {detection['ymin']:.0f}), ({detection['xmax']:.0f}, {detection['ymax']:.0f})\n")
                f.write("-----------------------\n")
                
                # Draw rectangle on image
                start_point = (int(detection['xmin']), int(detection['ymin']))
                end_point = (int(detection['xmax']), int(detection['ymax']))
                color = (0, 255, 0)
                thickness = 2
                original_image = cv2.rectangle(original_image, start_point, end_point, color, thickness)
                
                # Add label
                label = f"{detection['name']} {detection['confidence']:.2f}"
                cv2.putText(original_image, label, (start_point[0], start_point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # Save annotated image
        output_image_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_detected.jpg")
        cv2.imwrite(output_image_path, original_image)

    except Exception as e:
        print(f"Error processing {image_file}: {str(e)}")

print(f"\nProcessing complete! Results saved in '{output_dir}' directory")

# Create summary file
summary_path = os.path.join(output_dir, "processing_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"Total images processed: {num_images}\n")
    f.write(f"Input directory: {input_dir}\n")
    f.write(f"Output directory: {output_dir}\n")
    f.write("\nProcessed images:\n")
    for img in selected_images:
        f.write(f"- {img}\n")