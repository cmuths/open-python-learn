import cv2
import numpy as np

# 读取图像
img = cv2.imread('ocr/1-1.png')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用自适应阈值法进行二值化处理
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 应用形态学操作（腐蚀和膨胀）
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(thresh, kernel)
dilated = cv2.dilate(eroded, kernel)

# 合并腐蚀和膨胀结果
merged = cv2.bitwise_or(eroded, dilated)

# 应用双边滤波器进行平滑处理
filtered = cv2.bilateralFilter(merged, 9, 75, 75)

# 显示结果
cv2.imshow('Output Image', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()