import numpy as np
import cv2


img = np.zeros((400, 600, 3),  np.uint8)

img[:] = (250, 182, 236)


photo = cv2.imread("images/resume 2.jpg")
photo = cv2.resize(photo, (120, 120))
img[20:140, 20:140] = photo

cv2.rectangle(img, (5, 5), (595, 395), (197, 161, 247), 4)



cv2.putText(img, "Andrii Lishchenko", (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
cv2.putText(img, "Python Developer", (160, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2)
cv2.putText(img, "Email: andrii.lishchenko1@gmail.com", (160, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(img, "Phone: +380689700997", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(img, "Birth: 17.10.1937", (160, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(img, "OpenCV Business Card", (80, 334), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 0, 120), 2)

photo = cv2.imread("images/resume 2.jpg")

qr = cv2.imread("images/qr.png")
qr = cv2.resize(qr, (100, 100))
img[250:350, 450:550] = qr


cv2.imwrite("business_card.png", img)

cv2.imshow("Business Card", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

