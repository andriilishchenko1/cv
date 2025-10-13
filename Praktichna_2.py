import cv2
import numpy as np


img = cv2.imread('images/figurez.png')
img_copy = img.copy()

img = cv2.GaussianBlur(img,(5,5),2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


lower = np.array([0, 42, 0])
upper = np.array([179, 255, 255])
mask = cv2.inRange(img, lower, upper)
img = cv2.bitwise_and(img_copy, img, mask = mask)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = lower = np.array([40, 0, 0])
upper_green = np.array([58, 255, 255])



lower_red = lower = np.array([160, 0, 0])
upper_red = np.array([179, 255, 255])


lower_blue = lower = np.array([85, 36, 0])
upper_blue = np.array([153, 255, 255])


mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_green = cv2.inRange(hsv, lower_green, upper_green)

#mask_total = cv2.bitwise_or(mask_red, mask_blue)
#mask_total = cv2.bitwise_or(mask_total, mask_green)


colors = {"red": mask_red, "blue": mask_blue, "green": mask_green}








contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)#площа фігури
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)

        perimeter = cv2.arcLength(cnt, True)# perimetr kontura
        cv2.putText(img_copy, f'S:{int(area)}, P:{perimeter}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        M = cv2.moments(cnt)# M - моменти контуру

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            aspect_ratio = round(w / h, 2 )
            compact = round((4 * np.pi * area) / (perimeter ** 2), 2)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) == 4:
                shape = "sq"
            if len(approx) < 8:
                shape = "papugai"
            else:
                shape = "elipse"
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (138, 15, 96), 2)
        cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img_copy, f'X:{x}, Y:{y}', (x, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compact}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img_copy, f'shape:{shape}', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img_copy, f'colour:{colors}', (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('image', img)
cv2.imshow('mask', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()