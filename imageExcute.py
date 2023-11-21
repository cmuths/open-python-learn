import math
import sys

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from util.opencvUtilSharpen import Sharpen as sharpen

imgs = []
titles = []
threadCount = 2
# 原图片读取
sim = cv.imread('ocr/20231014-101559.jpg')


imgs.append(cv.cvtColor(sim, cv.COLOR_BGR2RGB))
titles.append('source')

fastnose = cv.fastNlMeansDenoisingColored(sim, h=10, templateWindowSize=7, searchWindowSize=21)
imgs.append(cv.cvtColor(fastnose, cv.COLOR_BGR2RGB))
titles.append('fastnose')

# kernel = np.ones((3,3),np.uint8)
# fastnose = cv.erode(fastnose, kernel, iterations=2)

# upim = cv.convertScaleAbs(fastnose,alpha=1.3,beta=0)
upim=cv.normalize(fastnose,dst=None,alpha=250,beta=10,norm_type=cv.NORM_MINMAX)
imgs.append(cv.cvtColor(upim, cv.COLOR_BGR2RGB))
titles.append('upim')





sliptim = upim

# 通道分离
b, g, r, a = None, None, None, None
if sliptim is not None and len(sliptim.shape) == 3:  # 彩色图像才可以做通道分离
    if sliptim.shape[2] == 3:  # 如果是3通道，分离出3个图像实例
        b, g, r = cv.split(sliptim)
    elif sliptim.shape[2] == 4:  # 如果是4通道
        b, g, r, a = cv.split(sliptim)

if a is None:
    for x in [b, g, r]:
        # x = cv.fastNlMeansDenoising(x, h=10, templateWindowSize=7, searchWindowSize=21)
        imgs.append(x)
        lower_gray = 210
        upper_gray = 255
        mask = cv.inRange(x, lower_gray, upper_gray)
        x = cv.bitwise_and(x, x, mask=mask)
        imgs.append(mask)
    for x in ['b', 'g', 'r']:
        titles.append(x)
        titles.append(x+'-gray')
else:
    for x in [b, g, r, a]:
        imgs.append(x)
    for x in ['b', 'g', 'r', 'a']:
        titles.append(x)

# 阈值过滤

for i in range(len(imgs)):
    if i <= threadCount:
        continue
    # ret, binaryIm = cv.threshold(imgs[i], 127, 255, cv.THRESH_BINARY)
    ret, tozeroIm = cv.threshold(imgs[i], 205, 255, cv.THRESH_TOZERO)
    for x in [tozeroIm]:
        imgs.append(x)
    for x in ['tozeroIm']:
        titles.append(titles[i]+'-'+x)

for i in range(len(imgs)):
    if i <= threadCount:
        continue
    cv.imwrite('ocr/'+titles[1]+'.png',imgs[i])


plt.figure(figsize=(30,20))
for i in range(len(imgs)):
    plt.subplot(math.ceil(len(imgs)/3), 3, i + 1), plt.imshow(imgs[i], 'gray')
    plt.ylabel(titles[i],fontdict={'weight':'bold','size': 20})
    plt.xticks([]), plt.yticks([])
plt.show()
