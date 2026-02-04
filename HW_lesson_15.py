import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, "video")
OUT_DIR = os.path.join(PROJECT_DIR, "out")

os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, "video.mp4")
    cap = cv2.VideoCapture(VIDEO_PATH)

model = YOLO('yolov8n.pt')

CONF_TRESHOLD = 0.4

RESIZE_WIDTH = 960

prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    result = model.predict(frame, conf=CONF_TRESHOLD, verbose=False)

    cats_count = 0
    dogs_count = 0

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 15:
                cats_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Cat: {conf:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elif cls == 16:
                dogs_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Dog: {conf:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    total_animals = cats_count + dogs_count

    now = time.time()
    dt = now - prev_time
    prev_time = now

    if dt > 0:
        fps = 1.0 / (dt)

    cv2.putText(frame, f"Cats: {cats_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Dogs: {dogs_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Total: {total_animals}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()