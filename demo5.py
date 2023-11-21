import cv2
import numpy as np

# 读取图像
img = cv2.imread('ocr/1-5.png')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 应用形态学操作（腐蚀和膨胀）
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(blurred, kernel)
dilated = cv2.dilate(eroded, kernel)

# 创建掩膜
mask = np.zeros_like(img)
mask[dilated == 255] = 1

# 应用掩膜
output = cv2.bitwise_and(img, mask)

# 显示结果
cv2.imshow('Output Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()