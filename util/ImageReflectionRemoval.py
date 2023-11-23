import cv2
import numpy as np


class ImageReflectionRemoval:
    def remove_reflection_histogram(self, img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 计算梯度方向直方图
        hist = cv2.calcHist([gray], [0], None, [180], [0, 256])
        # 归一化直方图
        # hist = cv2.normalize(hist, hist).astype(np.float32)
        # 应用直方图到原始图像上
        img_refined = cv2.LUT(gray, hist)

        return img_refined

    def remove_reflection_adaptive_threshold(self, img):
        # 计算图像的梯度幅值和方向
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # 计算梯度的幅度和方向
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        # 根据梯度的幅度和方向进行自适应阈值处理
        _, thresholded = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 根据梯度的方向进行图像融合
        kernel = np.ones((3, 3), np.uint8)
        img_refined = cv2.filter2D(img, -1, kernel) * thresholded / 255 + img * (1 - thresholded / 255)
        return img_refined

    def remove_reflection_reflection_model(self, img):
        # 计算图像的梯度幅值和方向
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # 计算梯度的幅度和方向
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        # 根据梯度的幅度和方向进行图像融合
        kernel = np.ones((3, 3), np.uint8)
        img_refined = cv2.filter2D(img, -1, kernel) * magnitude / 255 + img * (1 - magnitude / 255)
        return img_refined