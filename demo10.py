import cv2
import numpy as np

# 加载图像
img = cv2.imread('ocr/1-1.png')

# 将图像转换为HSV空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义你想要模糊的颜色在HSV空间的范围（这里是绿色）
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

# 在这个颜色范围内应用模糊
mask = cv2.inRange(hsv, lower_red, upper_red)
blurred = cv2.GaussianBlur(img, (15, 15), 0)  # 使用高斯模糊进行模糊处理
blurred_mask = cv2.bitwise_and(blurred, blurred, mask=mask)

# 显示原始图像、模糊的图像和模糊特定颜色的图像
cv2.imshow('Original Image', img)
cv2.imshow('Blurred Image', blurred)
cv2.imshow('Blurred Green', blurred_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()