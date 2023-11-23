import cv2
import numpy as np


def compare_histograms(image1, image2):
    # 计算图像的直方图
    hist_1, _ = np.histogram(image1.flatten(), 256, [0, 256])
    hist_2, _ = np.histogram(image2.flatten(), 256, [0, 256])

    # 计算直方图的归一化版本
    hist_1 = hist_1.astype("float") / hist_1.max()
    hist_2 = hist_2.astype("float") / hist_2.max()

    # 计算直方图的相似度
    # 使用Chi-Squared距离作为相似度度量
    dissimilarity = np.sum((hist_1 - hist_2) ** 2)
    return dissimilarity


# 加载图像
image1 = cv2.imread("ocr/3-2.png")
image2 = cv2.imread("ocr/6-1.png")

# 将图像转换为灰度图像（如果它们不是灰度的）
if len(image1.shape) > 2:
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
if len(image2.shape) > 2:
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 比较直方图并打印相似度分数
similarity = compare_histograms(image1, image2)
print(f"Histogram Similarity Score: {similarity}")