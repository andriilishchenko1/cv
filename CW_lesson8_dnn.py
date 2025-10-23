import cv2
import numpy as np

face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]#масшабування коорд. обличчя

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()
    #print(detections.shape)


    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2 ,y2 ) = box.astype("int")

            x, y = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            roi_gray = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
            roi_color = frame[y:y2, x:x2]

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10))
            for (xe, ye, we, he) in eyes:
                cv2.rectangle(roi_color, (xe, ye), (xe+ we, ye + he), (0, 0, 255))

            smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=10, minSize=(25, 25))
            for (xs, ys, ws, hs) in smile:
                cv2.rectangle(roi_color, (xs, ys), (xs + ws, ys + hs), (0, 255, 255))



    cv2.imshow('vid',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

