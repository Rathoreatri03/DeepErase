import torch
import cv2
import numpy as np

class LogoDetection:
    def __init__(self, model_path, video_path, output_path, detections_path, conf_threshold=0.6, iou_threshold=0.45):
        # Initialize paths and model parameters
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.detections_path = detections_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.cap = None
        self.fps = None
        self.frame_width = None
        self.frame_height = None

        # Load YOLOv5 model
        self.load_model()

    def load_model(self):
        """Load YOLOv5 custom model."""
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=self.model_path, force_reload=True)
        self.model.conf = self.conf_threshold  # Confidence threshold
        self.model.iou = self.iou_threshold   # NMS IOU threshold
        self.model.classes = None  # Detect all classes
        self.model.eval()

    def process_video(self):
        """Process the video, detect logos, and yield frame-by-frame detections."""
        # Open Video File
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("❌ Error: Could not open input video.")
            return []

        # Get Video Properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternative: 'MJPG'
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

        # Open a text file for storing detections
        with open(self.detections_path, "w") as file:
            file.write("Frame,Class,Confidence,Xmin,Ymin,Xmax,Ymax\n")  # CSV Header
            frame_count = 0

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("✅ Video processing complete.")
                    break

                frame_count += 1
                print(f"📽 Processing frame {frame_count}...")

                # Convert BGR to RGB for YOLOv5
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform Object Detection
                results = self.model(rgb_frame)
                detections = results.pandas().xyxy[0]  # Get detections for the frame

                # Prepare a list for storing frame-level detection information
                frame_detections = []

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

                    # Store frame-specific detection info
                    frame_detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [xmin, ymin, xmax, ymax]
                    })

                # Write Processed Frame to Output Video
                out.write(frame)

                # Yield the detections for the current frame
                yield frame_count, frame_detections

        self.cap.release()
        out.release()

# Example Usage
if __name__ == "__main__":
    model_path = r"E:\Stream_Censor\DeepErase\model\openlogo.pt"
    video_path = r"E:\Stream_Censor\DeepErase\assets\datasets\puma.mp4"
    output_path = r"/DeepErase/results\output_video.mp4"
    detections_path = r"/DeepErase/results\detections.txt"

    logo_detector = LogoDetection(model_path, video_path, output_path, detections_path)

    # Process video and get frame-wise detection results
    for frame_number, frame_detections in logo_detector.process_video():
        print(f"Frame {frame_number} Detections: {frame_detections}")
        # You can now process or store frame_detections as needed
