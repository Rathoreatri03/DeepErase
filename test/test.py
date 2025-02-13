import torch
import cv2
import os
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
from tqdm import tqdm


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
def video_analysis(video_path, bbox_file, output_frame_path, output_video_path, pipe):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Read bbox data
    bbox_frames = {}
    if os.path.exists(bbox_file):
        with open(bbox_file, "r") as f:
            for line in f:
                frame_num = int(line.split()[0])
                if frame_num in bbox_frames:
                    bbox_frames[frame_num].append(line)
                else:
                    bbox_frames[frame_num] = [line]

    frame_count = 0
    progress_bar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = frame.copy()  # Default to original frame

        # If this frame has bounding boxes, process it
        if frame_count in bbox_frames:
            # Convert frame to RGB for processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with all its bounding boxes
            for bbox_line in bbox_frames[frame_count]:
                result = process_image(image, bbox_line, pipe)
                if result is not None:
                    # Convert PIL Image back to OpenCV format
                    output_frame = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        # Save individual frame
        frame_filename = os.path.join(output_frame_path, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, output_frame)

        # Write frame to video
        video_writer.write(output_frame)

        frame_count += 1
        progress_bar.update(1)

    # Cleanup
    progress_bar.close()
    cap.release()
    video_writer.release()
    torch.cuda.empty_cache()


def process_image(frame, line, pipe):
    """
    Process a single frame with Stable Diffusion inpainting.
    Returns PIL Image.
    """
    # Convert numpy array to PIL Image
    image = Image.fromarray(frame)
    img_width, img_height = image.size

    # Create mask
    mask = Image.new("L", (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)

    # Parse bbox and draw on mask
    bbox = parse_yolo_bbox(line, img_width, img_height)
    if bbox:
        x1, y1, x2, y2, _ = bbox
        draw.rectangle([x1, y1, x2, y2], fill=255)

    # Resize for Stable Diffusion
    image_resized = image.resize((512, 512))
    mask_resized = mask.resize((512, 512))

    # Perform inpainting
    result = pipe(
        prompt="natural background scene, high quality, photorealistic",
        negative_prompt="text, watermark, logo, artificial, distorted",
        image=image_resized,
        mask_image=mask_resized,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    # Resize back to original size
    return result.resize(image.size)


def main():
    video_path = r"E:\Stream_Censor\StreamClear\assets\datasets\videoplayback.mp4"
    bbox_file = r"E:\Stream_Censor\StreamClear\test\detection_results\output_labels.txt"
    output_frame_path = r"E:\Stream_Censor\StreamClear\test\detection_results\frames"
    output_video_path = r"E:\Stream_Censor\StreamClear\test\detection_results\output_video.mp4"

    # Create output directory if it doesn't exist
    os.makedirs(output_frame_path, exist_ok=True)

    # Load Stable Diffusion Model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    try:
        print("Starting video processing...")
        video_analysis(video_path, bbox_file, output_frame_path, output_video_path, pipe)
        print(f"Processing complete. Output video saved to: {output_video_path}")
        print(f"Individual frames saved to: {output_frame_path}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()