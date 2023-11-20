import cv2 as cv;
from matplotlib import pyplot as plt

imgs = []
titles = []
# 原图片读取
sim = cv.imread('ocr/1-1.png')

imgs.append(cv.cvtColor(sim, cv.COLOR_BGR2RGB))
titles.append('source')

# 降噪，中值滤波
# blimg = cv.medianBlur(sim, 7)
# imgs.append(cv.cvtColor(blimg, cv.COLOR_BGR2RGB))
# titles.append('blur')

# 通道分离
b, g, r, a = None, None, None, None
if sim is not None and len(sim.shape) == 3:  # 彩色图像才可以做通道分离
    if sim.shape[2] == 3:  # 如果是3通道，分离出3个图像实例
        b, g, r = cv.split(sim)
    elif sim.shape[2] == 4:  # 如果是4通道
        b, g, r, a = cv.split(sim)

if a is None:
    for x in [b, g, r]:
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
    if i <= 1:
        continue
    imt = cv.medianBlur(imgs[i],5)
    ret, binaryIm = cv.threshold(imt, 127, 255, cv.THRESH_BINARY)
    ret, tozeroIm = cv.threshold(imt, 127, 255, cv.THRESH_TOZERO)
    for x in [binaryIm,tozeroIm]:
        imgs.append(x)
    for x in ['binaryIm','tozeroIm']:
        titles.append(x)

plt.figure(figsize=(10,100))
for i in range(len(imgs)):
    plt.subplot(len(imgs), 1, i + 1), plt.imshow(imgs[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
