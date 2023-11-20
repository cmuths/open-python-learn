import cv2 as cv
from matplotlib import pyplot as plt

def adaptive_threshold():
    img = cv.imread('ocr/1-1.png')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 中值滤波
    img_gray = cv.medianBlur(img_gray, 5)
    ret, th1 = cv.threshold(img_gray, 97, 255, cv.THRESH_BINARY)
    # 11 为 Block size,即邻域大小，用于计算阈值的窗口大小, 2 为 常数，可以理解为偏移量
    th2 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding',
              'Adaptive Gaussian Thresholding']
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    images = [img_rgb, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

adaptive_threshold()