import cv2
from ultralytics import YOLO



model = YOLO("yolov8m.pt")  
CONF_THRESHOLD = 0.7


WASTE_MAP = {
    "bottle": ("GLASS BIN", (0, 255, 0)),
    "wine glass": ("GLASS BIN", (0, 255, 0)),
    "cup": ("PLASTIC BIN", (0, 255, 255)),
    "book": ("PAPER BIN", (255, 200, 0)),
    "cell phone": ("E-WASTE", (255, 0, 255)),
}

cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD, imgsz=960, verbose=False)

    recommendation_text = "NOT DETECTED"
    recommendation_color = (0, 0, 255)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name in WASTE_MAP:
                recommendation_text, recommendation_color = WASTE_MAP[class_name]

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              recommendation_color, 3)

                cv2.putText(frame,
                            f"{class_name} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            recommendation_color,
                            2)

    
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)

    cv2.putText(frame,
                "RECOMMENDED BIN:",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                recommendation_text,
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                recommendation_color,
                3)

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("PRoject", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

