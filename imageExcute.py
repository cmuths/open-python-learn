import math
import sys

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from util.opencvUtilSharpen import Sharpen as sharpen

imgs = []
titles = []
threadCount = 5
# 原图片读取
sim = cv.imread('ocr/1-1.png')


imgs.append(cv.cvtColor(sim, cv.COLOR_BGR2RGB))
titles.append('source')

fastnose = cv.fastNlMeansDenoisingColored(sim, h=15, templateWindowSize=7, searchWindowSize=21)
imgs.append(cv.cvtColor(fastnose, cv.COLOR_BGR2RGB))
titles.append('fastnose')

filtered_img = cv.bilateralFilter(sim, 9, 75, 75)
imgs.append(cv.cvtColor(filtered_img, cv.COLOR_BGR2RGB))
titles.append('filtered_img')

# 降噪，中值滤波
blimg = cv.medianBlur(sim, 5)
imgs.append(cv.cvtColor(blimg, cv.COLOR_BGR2RGB))
titles.append('noise-medianBlur')


#锐化

usm_sharpen = sharpen.unsharp_masking(sim)
imgs.append(cv.cvtColor(usm_sharpen, cv.COLOR_BGR2RGB))
titles.append('usm_sharpen')

hpf_sharpen = sharpen.high_pass_filter(fastnose)
imgs.append(cv.cvtColor(hpf_sharpen, cv.COLOR_BGR2RGB))
titles.append('hpf_sharpen')



sliptim = usm_sharpen

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
        x = cv.medianBlur(x,5)
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
    # ret, binaryIm = cv.threshold(imgs[i], 127, 255, cv.THRESH_BINARY)
    ret, tozeroIm = cv.threshold(imgs[i], 127, 255, cv.THRESH_TOZERO)
    for x in [tozeroIm]:
        imgs.append(x)
    for x in ['tozeroIm']:
        titles.append(titles[i]+'-'+x)

for i in range(len(imgs)):
    if i <= threadCount:
        continue
    cv.imwrite('ocr/'+titles[1]+'.png',imgs[i])


plt.figure(figsize=(30,50))
for i in range(len(imgs)):
    plt.subplot(math.ceil(len(imgs)/3), 3, i + 1), plt.imshow(imgs[i], 'gray')
    plt.ylabel(titles[i],fontdict={'weight':'bold','size': 20})
    plt.xticks([]), plt.yticks([])
plt.show()
