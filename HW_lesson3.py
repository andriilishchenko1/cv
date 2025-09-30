import numpy as np
import cv2
img = cv2.imread('2.png')

cv2.rectangle(img, (80, 30), (260, 315), (0, 0, 255), 2)
cv2.putText(img, "Lishchenko Andrii", (30,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('img',img)
print(img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()