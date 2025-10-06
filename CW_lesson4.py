import cv2
import numpy as np
img = cv2.imread('123.jpg')
scale = 1
img = cv2.resize(img,(img.shape[1]//scale,img.shape[0]//scale))
print(img.shape)

img_copy = img.copy()

img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

img_copy = cv2.GaussianBlur(img_copy,(5,5),2)

img_copy = cv2.equalizeHist(img_copy)#Посилаення контрасту

img_copy = cv2.Canny(img_copy, 70, 50)
img_copy_color = img.copy()

#пошук контурів
contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#1 - режим, який отримує зовнішні контури, retr external знаходить крайній контур, якщо об'єкт має дюрку, то дюрка буде ігноруватся
#2 малювання контурів, прямокутників та тексту

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2)#Список конутрів, які ми малюємо
        cv2.rectangle(img_copy_color, (x,y), (x + w, y + h), (0, 255, 0), 2)
    text_y = y-5 if y - 5 > 10 else y + 15
    text = f'x:{x}  y:{y}  s:{int(area)}  h:{h}'
    cv2.pydraw.putText(img_copy_color, text, (x,text_y), (255,255,255), 2)



cv2.imshow('Copy', img_copy)
cv2.imshow('Canny', img_copy_color)
cv2.imshow('Kontyru i koordunatu', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

