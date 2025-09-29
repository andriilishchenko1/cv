import cv2


img1 = cv2.imread("images/1.jpg")
img2 = cv2.imread("images/2.png")


img1_resized = cv2.resize(img1, (1000, 300))
img2_resized = cv2.resize(img2, (300, 400))


gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)


edges1 = cv2.Canny(gray1, 100, 200)
edges2 = cv2.Canny(gray2, 80, 100)


cv2.imshow("Image 1 - edges", edges1)
cv2.imshow("Image 2 - edges", edges2)



cv2.waitKey(0)
cv2.destroyAllWindows()
