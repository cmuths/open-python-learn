import cv2 as cv
import numpy as np


class Thresh:
    def adaptiveThreshold(img):
        max_kernel_size = 5
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, max_kernel_size,10)

    def bimodalThreshold(img):
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        T = np.where(hist == max(hist[:-1]))[0][0] + 1  # 在双峰之间的波谷所代表的灰度值T处设置阈值，使得背景与目标被分割开来。    # 使用max找到第一个峰值，然后在其左侧找到一个点，使得左侧的峰与右侧的峰的高度相同，这个点就是阈值T。注意，由于峰值是在左侧找到的，因此阈值T应该大于等于0。+1是为了确保阈值T大于等于0。
        binary_img = np.zeros_like(img)
        binary_img[img > T] = 255
        binary_img[img <= T] = 0
        return binary_img

    def OtsuThreshod(img):
        return  cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

