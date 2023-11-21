import cv2
import numpy as np

# 加载灰度图像
image = cv2.imread('ocr/1-1.png', cv2.IMREAD_GRAYSCALE)

# 定义需要去除的颜色范围（这里是0-100的蓝色）
lower_blue = np.array([0, 50, 50])
upper_blue = np.array([10, 255, 255])

# 创建掩膜，将蓝色像素设置为黑色（0），其他像素设置为白色（255）
mask = cv2.inRange(image, lower_blue, upper_blue)

# 使用掩膜去除特定颜色
result = cv2.bitwise_and(image, image, mask=mask)

# 显示原始图像和去除颜色的结果
cv2.imshow('Original Image', image)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()