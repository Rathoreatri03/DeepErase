import torch
import cv2
import os
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
from tqdm import tqdm


def video_analysis(video_path, bbox_file, output_image_path, pipe):
    cap = cv2.VideoCapture(video_path)

    # Create video writer
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_image_path, fourcc, fps, (width, height))
    progress_bar = tqdm(total=total_frames, desc="Processing frames")

    # Check if video is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_count = 0

    with open(bbox_file, "r") as f:
        bbox_file_data = f.readlines()  # Read all lines from the file

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing complete.")
            break

        frame_count += 1

        for line in bbox_file_data:
            frame_number = int(line.split()[0])  # Extract frame number from bbox line

            if frame_count == frame_number:  # Process only if it matches the current frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

                # Process the frame using the corresponding bbox information
                frame = process_image(image, line, output_image_path, pipe)
                output_file = os.path.join(output_image_path,
                                           f"inpainted_result_{frame_count}.png")  # Save as a new file
                Image.fromarray(frame).save(output_file)
                break  # No need to check other lines once matched

        output_file = os.path.join(output_image_path, f"inpainted_result_{frame_count}.png")  # Save as a new file
        Image.fromarray(frame).save(output_file)

        video_writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        progress_bar.update(1)

    cap.release()
    video_writer.release()
    progress_bar.close()


def parse_yolo_bbox(line, img_width, img_height):
    """
    Parse YOLO formatted bounding box (normalized) and convert to pixel coordinates.
    """
    values = line.strip().split()

    frame, _, x_center, y_center, bbox_width, bbox_height = map(float, values)

    # Convert from normalized to pixel coordinates
    x1 = int((x_center - bbox_width / 2) * img_width)
    y1 = int((y_center - bbox_height / 2) * img_height)
    x2 = int((x_center + bbox_width / 2) * img_width)
    y2 = int((y_center + bbox_height / 2) * img_height)

    return x1, y1, x2, y2, frame


def process_image(frame, line, output_image_path, pipe):
    """
    Loads an image, applies a mask for detected logos, and uses Stable Diffusion inpainting.
    """
    # Load input image
    image = Image.fromarray(frame).convert("RGB")
    img_width, img_height = image.size

    # Create a blank mask (same size as input image)
    mask = Image.new("L", (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)

    bbox = parse_yolo_bbox(line, img_width, img_height)
    if bbox:
        x1, y1, x2, y2, frame = bbox
        draw.rectangle([x1, y1, x2, y2], fill=255)  # White mask on detected region

    # Resize to 512x512 for Stable Diffusion
    image_resized = image.resize((512, 512))
    mask_resized = mask.resize((512, 512))

    # Perform inpainting using Stable Diffusion
    result = pipe(
        prompt="natural background scene, high quality, photorealistic",
        negative_prompt="text, watermark, logo, artificial, distorted",
        image=image_resized,
        mask_image=mask_resized,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    # Resize back to original image size
    result = result.resize(image.size)

    # Convert to NumPy array for further processing
    result_np = np.array(result)

    # Convert NumPy array back to PIL Image for saving
    result_image = Image.fromarray(result_np)

    # Now save the inpainted image as a PNG file
    output_file = os.path.join(output_image_path, f"inpainted_result.png")
    result_image.save(output_file)  # Save the image as PNG

    return result_np  # Return NumPy array for further processing if needed


def main():
    video_path = r"E:\Stream_Censor\StreamClear\assets\datasets\videoplayback.mp4"
    bbox_file = r"E:\Stream_Censor\StreamClear\test\detection_results\output_labels.txt"
    output_image_path = r"E:\Stream_Censor\StreamClear\test\detection_results\output_inpainted"

    # Load Stable Diffusion Inpainting Model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    try:
        print("Processing video frames...")
        video_analysis(video_path, bbox_file, output_image_path, pipe)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
