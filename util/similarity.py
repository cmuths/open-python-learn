import cv2
import numpy as np
# from scipy import signal


class ImageSimilarity:
    def compare_histograms(self, image1, image2):
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

    def compare_ssim(self, image1, image2):
        # 计算结构相似性指数并计算相似度分数
        similarity = cv2.compareSSIM(image1, image2, win_size=3)
        return similarity

    def compare_features(self, image1, image2):
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
        return similarity

    def compare_mse(self, image1, image2):
        # 计算两个图像的平均差异值
        mse = np.mean((image1 - image2) ** 2)
        return mse

    # def compare_psnr(self, image1, image2):
    #     # 计算两个图像的峰值信噪比
    #     psnr = signal.peak_snr(image1, image2)
    #     return psnr