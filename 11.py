import cv2
import numpy as np

# 读取两个图像
image1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 计算两个图像的直方图
hist_1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
hist_2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

# 比较直方图并计算相似度分数
similarity = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CORREL)
print(f"Histogram Similarity Score: {similarity}")

import cv2
import numpy as np

# 读取两个图像
image1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 计算结构相似性指数并计算相似度分数
similarity = cv2.compareSSIM(image1, image2, win_size=3)
print(f"SSIM Similarity Score: {similarity}")

import cv2
import numpy as np

# 读取两个图像并提取特征点
image1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
detector = cv2.SIFT_create()
keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

# 匹配特征点并计算相似度分数
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
similarity = len(np.where(np.array(matches[0, :]) < 0)[0]) / len(matches[0, :])
print(f"Feature Matching Similarity Score: {similarity}")

import cv2
import numpy as np

# 读取两个图像
image1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 计算两个图像的平均差异值
mse = np.mean((image1 - image2) ** 2)
print(f"MSE Similarity Score: {mse}")

import cv2
import numpy as np
from scipy import signal

# 读取两个图像
image1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 计算两个图像的峰值信噪比
psnr = signal.peak_snr(image1, image2)
print(f"PSNR Similarity Score: {psnr}")