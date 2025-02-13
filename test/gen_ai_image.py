import torch
import cv2
import os
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw

def parse_yolo_bbox(line, img_width, img_height):
    """
    Parse YOLO formatted bounding box (normalized) and convert to pixel coordinates.
    """
    values = line.strip().split()
    if len(values) != 5:
        return None  # Invalid format

    _, x_center, y_center, bbox_width, bbox_height = map(float, values)

    # Convert from normalized to pixel coordinates
    x1 = int((x_center - bbox_width / 2) * img_width)
    y1 = int((y_center - bbox_height / 2) * img_height)
    x2 = int((x_center + bbox_width / 2) * img_width)
    y2 = int((y_center + bbox_height / 2) * img_height)

    return x1, y1, x2, y2

def process_image(input_image_path, bbox_file, output_image_path, pipe):
    """
    Loads an image, applies a mask for detected logos, and uses Stable Diffusion inpainting.
    """
    # Load input image
    image = Image.open(input_image_path).convert("RGB")
    img_width, img_height = image.size

    # Create a blank mask (same size as input image)
    mask = Image.new("L", (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)

    # Read bounding box coordinates from YOLO format file
    with open(bbox_file, "r") as f:
        bbox_lines = f.readlines()

    for line in bbox_lines:
        bbox = parse_yolo_bbox(line, img_width, img_height)
        if bbox:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], fill=255)  # White mask on detected region

    # Resize to 512x512 for Stable Diffusion
    image_resized = image.resize((512, 512))
    mask_resized = mask.resize((512, 512))

    # Perform inpainting using Stable Diffusion
    result = pipe(
        prompt="background",
        image=image_resized,
        mask_image=mask_resized,
    ).images[0]

    # Resize back to original image size
    result = result.resize(image.size)

    # Save output
    result.save(output_image_path)
    print(f"Processed image saved at: {output_image_path}")

def main():
    input_image_path = r"E:\Stream_Censor\StreamClear\test\detection_results_image\original_frames\frame_0011.jpg"
    bbox_file = r"E:\Stream_Censor\StreamClear\test\detection_results_image\labels\frame_0011.txt"
    output_image_path = r"E:\Stream_Censor\StreamClear\test\detection_results_image\output_inpainted.jpg"

    # Load Stable Diffusion Inpainting Model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    try:
        print("Processing single image...")
        process_image(input_image_path, bbox_file, output_image_path, pipe)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
