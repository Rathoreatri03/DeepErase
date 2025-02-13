import cv2
import numpy as np
from PIL import Image, ImageDraw

def parse_bbox_line(line):
    """Parse bounding box coordinates from a text file."""
    try:
        values = [int(x) for x in line.strip().split()]
        return values[1], values[2], values[3], values[4]  # Adjust based on bbox format
    except:
        return None

def process_image_opencv(input_image_path, bbox_file, output_image_path):
    """Removes detected logos using OpenCV inpainting."""
    # Load input image
    image = cv2.imread(input_image_path)

    # Create a mask (same size as input image, single channel)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Read bounding box coordinates from file
    with open(bbox_file, "r") as f:
        bbox_lines = f.readlines()

    for line in bbox_lines:
        bbox = parse_bbox_line(line)
        if bbox:
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 255  # Mark detected region for inpainting

    # Apply OpenCV inpainting (TELEA is good for natural textures)
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Save the output
    cv2.imwrite(output_image_path, inpainted_image)
    print(f"Processed image saved at: {output_image_path}")

def main():
    input_image_path = r"E:\Stream_Censor\StreamClear\test\detection_results_image\original_frames\frame_0011.jpg"
    bbox_file = r"E:\Stream_Censor\StreamClear\test\detection_results_image\labels\frame_0011.txt"
    output_image_path = r"E:\Stream_Censor\StreamClear\test\detection_results_image\output_inpainted_opencv.jpg"

    try:
        print("Processing image with OpenCV inpainting...")
        process_image_opencv(input_image_path, bbox_file, output_image_path)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
