import cv2

# 读取图像
img = cv2.imread('ocr/1-5.png')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用直方图均衡化
equ = cv2.equalizeHist(gray)

# 显示结果
cv2.imshow('Histogram Equalization', equ)
cv2.waitKey(0)
cv2.destroyAllWindows()