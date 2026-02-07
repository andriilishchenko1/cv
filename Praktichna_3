import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PROJECT_DIR, "video", "traffic.mp4")

CONF_THRESHOLD = 0.4
RESIZE_WIDTH = 960

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video not found")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    counts = {name: 0 for name in VEHICLE_CLASSES.values()}

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            label = VEHICLE_CLASSES[cls_id]
            counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,f"{label} {conf:.2f}",(x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)

    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    y = 30
    for name, count in counts.items():
        cv2.putText(frame, f"{name}: {count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 30

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Traffic detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
