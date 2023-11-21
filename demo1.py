import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./ocr/1-5.png',0)

# img = cv2.fastNlMeansDenoising(img1,None,10,10,7,21)
# 中值滤波
img = cv2.medianBlur(img, 5)
# img = cv2.GaussianBlur(img,(5,5),0)
# img = cv2.morphologyEx(img,cv2.MORPH_OPEN,(5,5))
# img = cv2.blur(img,(5,5))
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret1, th5 = cv2.threshold(img, 200, 255, cv2.THRESH_TOZERO)
th5 = 255 - th5

th5 = cv2.Laplacian(th5,cv2.CV_64F)
# th5 = cv2.dilate(th5,(5,5),1)
# th5 = cv2.bilateralFilter(th5,9,75,75)
# 11 为 Block size, 2 为 C 值R
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 21, 10)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            cv2.THRESH_BINARY, 21, 10)
# th6 = cv2.bilateralFilter(th3,9,75,75)
titles = ['Original Image', 'Global Thresholding (v = 127)','TOZERO',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1,th5, th2, th3]
for i in range(5):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
