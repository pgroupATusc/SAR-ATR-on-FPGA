import cv2

realimage = []

realimage.append(cv2.imread("pic/1-2S1.png"))
realimage.append(cv2.imread("pic/2-BRDM2.png"))
realimage.append(cv2.imread("pic/3-BTR70.png"))
realimage.append(cv2.imread("pic/4-T62.png"))
realimage.append(cv2.imread("pic/5-ZIL131.png"))
realimage.append(cv2.imread("pic/6-BMP2.png"))
realimage.append(cv2.imread("pic/7-BTR-60.png"))
realimage.append(cv2.imread("pic/8-D7.png"))
realimage.append(cv2.imread("pic/9-T72.png"))
realimage.append(cv2.imread("pic/10-ZSU234.png"))

cv2.imshow("1-2S1", realimage[1])

cv2.waitKey(0)