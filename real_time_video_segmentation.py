import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 segmentation model
# model = YOLO("yolov8n-seg.pt") # small model ~7M
model = YOLO("yolov8m-seg.pt")  # med model ~20M

# Open webcam (0 = default Mac camera)
cap = cv2.VideoCapture(0)

# Optional: reduce resolution for speed
resize_width, resize_height = 320, 320

start = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize for faster inference
    frame_resized = cv2.resize(frame, (resize_width, resize_height))

    # Run segmentation
    results = model(frame_resized, device="cpu")

    # Draw masks
    annotated_frame = results[0].plot()

    # Show live result
    cv2.imshow("YOLOv8 Live Segmentation", annotated_frame)

    frame_count += 1

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end = time.time()
print(f"FPS: {frame_count / (end - start):.2f}")

# Cleanup
cap.release()
cv2.destroyAllWindows()