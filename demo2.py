import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./ocr/1-1.png')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img3 = cv2.threshold(img, ret, 255, cv2.THRESH_TOZERO)


titles = ['Original Image', 'AfterImage','img3']
images = [cv2.cvtColor(img,cv2.COLOR_BGR2RGB), img3,img3]
plt.figure(figsize=(10,10))
for i in range(2):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
