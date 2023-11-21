import cv2
import numpy as np


class Sharpen:

    def unsharp_masking(img):
        # 转换为灰度图像
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 模糊图像
        blur = cv2.GaussianBlur(img, (1, 1), 25)

        # 非锐化掩蔽
        sharpened = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

        # sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        return sharpened


    def high_pass_filter(img):
        # 转换为灰度图像
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 模糊图像
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        # 高通滤波器增强边缘
        sharpened = cv2.addWeighted(img, 1, blur, -0.5, 0)

        # sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        return sharpened


    def gradient_operator(img):
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 梯度算子增强边缘
        # 使用Sobel算子计算X和Y方向梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx ** 2 + sobely ** 2)
        # gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

        return gradient


    def laplacian_filter(img):
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 拉普拉斯滤波器增强边缘
        laplacian = cv2.Laplacian(gray, 6)
        laplacian = np.uint8(np.absolute(laplacian))
        # laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

        return laplacian

    def retinex(image, sigma_list=[0.3, 0.5, 0.8]):
        # 高斯滤波
        blur = cv2.GaussianBlur(image, (15, 15), 0)
        # Retinex算法
        retinex = np.log10(image) - np.log10(blur)
        # 对数变换后的图像进行缩放和平移
        retinex = cv2.normalize(retinex, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # 根据sigma_list对图像进行高斯滤波
        for sigma in sigma_list:
            retinex = cv2.GaussianBlur(retinex, (0, 0), sigma)
            # 返回到原始尺寸并返回
        return cv2.GaussianBlur(retinex, (0, 0), sigma_list[-1])
