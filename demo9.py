import cv2
import numpy as np

# 读取图像
img = cv2.imread('ocr/1-1.png')

# 将图像转换为HSV颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义红光的色调H值范围
red_min = 0
red_max = 10

# 创建一个掩膜，将红光区域与非红光区域分离
mask = cv2.inRange(hsv, (red_min, 255, 255), (red_max, 255, 255))

# 将红光区域替换为白光
modified = np.copy(img)
modified[mask != 0] = [255, 255, 255]  # BGR格式的白光颜色

# 显示修改后的图像
cv2.imshow('Modified Image', modified)
cv2.waitKey(0)
cv2.destroyAllWindows()