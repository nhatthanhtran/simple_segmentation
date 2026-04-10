import cv2
from ultralytics import YOLO
import os
import time
import torch
# ----- CONFIG -----
video_path = "./data/video/"
video_name = "example1.mp4"
input_video = f"{video_path}{video_name}"      # path to input video
output_folder = "./results/video/"  # folder to save output video
output_video_name = video_name.replace(".mp4", "_segmented.mp4")  # output video name
resize_width, resize_height = 320, 680  # resize frames for faster processing
target_fps = 5
# ------------------

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, output_video_name)

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")
# a = torch.zeros((3,128,128))
# import pdb; pdb.set_trace()

# Open video
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
# fps = 8
frame_skip = max(int(fps // target_fps), 1)  # how many frames to skip


# Video writer (BGR frames)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (resize_width, resize_height))

frame_count = 0
processed_count = 0
start = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue
    # Resize frame
    frame_resized = cv2.resize(frame, (resize_width, resize_height))

    # Run YOLOv8 segmentation
    results = model(frame_resized, device="cpu")

    annotated_frame = results[0].plot()  # draw masks

    # Write annotated frame to video
    out.write(annotated_frame)

    processed_count += 1
    if processed_count % 50 == 0:
        print(f"Processed {processed_count} sampled frames...")

    frame_count += 1
print("VideoWriter opened:", out.isOpened())
cap.release()
out.release()
print(f"Segmented video saved at: {output_path}")
end = time.time()
print(f"Total processing time: {end - start:.2f} seconds")