import math
import sys

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from util.ImageReflectionRemoval import ImageReflectionRemoval
from util.opencvUtilSharpen import  Sharpen
from util.border import  Border

imgs = []
titles = []
threadCount = 2
threshValue = 127

# 原图片读取
sim = cv.imread('ocr/9-1.png')
imgs.append(cv.cvtColor(sim, cv.COLOR_BGR2RGB))
titles.append('source')


fastnose = cv.fastNlMeansDenoisingColored(sim, h=10, templateWindowSize=7, searchWindowSize=21)
imgs.append(cv.cvtColor(fastnose, cv.COLOR_BGR2RGB))
titles.append('fastnose')


upim=cv.normalize(fastnose,dst=None,alpha=250,beta=10,norm_type=cv.NORM_MINMAX)
imgs.append(cv.cvtColor(upim, cv.COLOR_BGR2RGB))
titles.append('upim')


border = Border()


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
    for x in ['b', 'g', 'r']:
        titles.append(x)
else:
    for x in [b, g, r, a]:
        imgs.append(x)
    for x in ['b', 'g', 'r', 'a']:
        titles.append(x)

# 阈值过滤

for i in range(len(imgs)):
    if i <= threadCount:
        continue
    ret, tozeroIm = cv.threshold(imgs[i], threshValue, 255, cv.THRESH_TOZERO)
    count = cv.countNonZero(tozeroIm)
    print(titles[i] + '-' + str(count))
    blurred = cv.GaussianBlur(tozeroIm, (7, 7), 0)
    tozeroIm = cv.addWeighted(tozeroIm, 1, blurred, 2, 0)
    filter = [0,1000000]

    # if threshValue >= 210:
    #     filter = [0,50000]
    # elif  threshValue <= 127 :
    #     filter = [100000,200000]
    # else:
    #     filter = [50000,100000]

    if filter[0] <= count <= filter[1]:
        for x in [tozeroIm]:
            imgs.append(x)
        for x in ['tozeroIm']:
            titles.append(titles[i]+'-'+x)

plt.figure(figsize=(30,20))
for i in range(len(imgs)):
    plt.subplot(math.ceil(len(imgs)/3), 3, i + 1), plt.imshow(imgs[i], 'gray')
    plt.ylabel(titles[i],fontdict={'weight':'bold','size': 20})
    plt.xticks([]), plt.yticks([])
plt.show()
