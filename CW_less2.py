import numpy as np
import cv2


img = np.zeros((500,400,3), np.uint8)
#img[:] = (155,235,225)
#rgb = bgr

#img[100:150, 200:250] = (155,235,225)

#cv2.rectangle(img,(100,100),(300,200),(155,235,225), 3)#1 координата, 2 координата, #3 колір, 4# товщина
#cv2.line(img, (100,100), (300,200), (155,235,225), 3)
#print(img.shape)
#cv2.line(img, (0, img.shape[0] // 2), (img.shape[1] ,img.shape[0] // 2), (155,235,225), 3)
#cv2.line(img, (img.shape[1] // 2, 0), (img.shape[1] // 2 ,img.shape[0]), (155,235,225), 3)
#cv2.circle(img, (200, 200), 100, (155,235,225), 3)
#cv2.putText(img, "Lishchenko Andrii", (90,150), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 3)
#
#cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
