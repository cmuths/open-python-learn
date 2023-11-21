import cv2
import numpy as np

def illum(img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_bw, 180, 255, 0)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    img_zero = np.zeros(img.shape, dtype=np.uint8)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        img_zero[y:y+h, x:x+w] = 255
    mask = img_zero
    result = cv2.illuminationChange(img, mask, alpha=1, beta=2)
    return result


def illuma(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(edges, kernel)
    dilated = cv2.dilate(eroded, kernel)
    mask = np.zeros_like(img)
    mask[edges == 255] = 1
    output = cv2.bitwise_and(img, mask)
    return output

# 读取原图像
img = cv2.imread("ocr/1-5.png")

img11 = illuma(img)
# 显示原图像和处理后的图像
cv2.imshow("Original", img)
cv2.imshow("Result", img11)
cv2.waitKey(0)
