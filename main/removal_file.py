import cv2
import numpy as np
import os
from PIL import Image
from core import process_inpaint

# Paths
video_path = r"E:\Stream_Censor\DeepErase\assets\datasets\puma.mp4"
detections_path = r"E:\Stream_Censor\DeepErase\assets\results\detections.txt"
output_video_path = r"E:\Stream_Censor\DeepErase\assets\results\inpaint_video.mp4"
frames_dir = r"E:\Stream_Censor\DeepErase\assets\results\processed_frames"

# Ensure output directory exists
os.makedirs(frames_dir, exist_ok=True)


def load_detections(detections_path):
    """Loads detections from the file."""
    detections = {}
    with open(detections_path, "r") as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split(",")
            frame_id = int(parts[0])
            xmin, ymin, xmax, ymax = map(int, parts[3:7])
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append((xmin, ymin, xmax, ymax))
    return detections


def create_mask(image, bounding_boxes):
    """Creates an RGBA mask where transparent areas (0,0,0,0) are inpainted."""
    img_array = np.array(image)
    h, w, _ = img_array.shape
    mask = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA mask

    # Detect magenta pixels (255, 0, 255)
    magenta_pixels = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 255)

    # Apply masking logic
    mask[:, :, :] = [0, 0, 0, 255]  # Default: Fully opaque
    mask[magenta_pixels] = [0, 0, 0, 0]  # Magenta areas become transparent

    # Apply bounding boxes as additional mask areas
    for (xmin, ymin, xmax, ymax) in bounding_boxes:
        mask[ymin:ymax, xmin:xmax] = [0, 0, 0, 0]  # Bounding box areas also transparent

    return mask


def apply_inpainting(image, bounding_boxes):
    """Applies inpainting to a frame."""
    img_input = Image.fromarray(image).convert("RGBA")
    mask = create_mask(img_input, bounding_boxes)

    # Process inpainting
    output = process_inpaint(np.array(img_input), mask)
    return output


def process_video(video_path, detections):
    """Processes video frame by frame."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id in detections:
            print(f"Processing frame {frame_id}")
            inpainted_frame = apply_inpainting(frame, detections[frame_id])
            frame_output_path = os.path.join(frames_dir, f"frame_{frame_id:04d}.png")
            cv2.imwrite(frame_output_path, inpainted_frame)
        else:
            frame_output_path = os.path.join(frames_dir, f"frame_{frame_id:04d}.png")
            cv2.imwrite(frame_output_path, frame)
            print(f"Skipping frame {frame_id}")

    cap.release()


def reconstruct_video(frames_dir, output_video_path, fps, frame_size):
    """Reconstructs video from processed frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    frame_files = sorted(os.listdir(frames_dir))
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print(f"Final video saved as {output_video_path}")


# Load detections
detections = load_detections(detections_path)

# Process video frames
process_video(video_path, detections)

# Reconstruct the video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()

reconstruct_video(frames_dir, output_video_path, fps, frame_size)
