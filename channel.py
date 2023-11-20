import cv2
from matplotlib import pyplot as plt

plt.figure(figsize=(16, 8))

img = cv2.imread('ocr/20231016-104440.jpg')
# img = cv2.imread('..\\opencv-logo.png',cv2.IMREAD_UNCHANGED)
if img is not None and len(img.shape) == 3:  # 彩色图像才可以做通道分离
    print('img.shape:', img.shape)
    if img.shape[2] == 3:  # 如果是3通道，分离出3个图像实例
        b, g, r = cv2.split(img)
        images = [b, g, r]
        titles = ['b', 'g', 'r']
        for i in range(3):
            plt.subplot(3, 1, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    elif img.shape[2] == 4:  # 如果是4通道
        b, g, r, a = cv2.split(img)
        images = [b, g, r, a]
        titles = ['b', 'g', 'r', 'a']
        for i in range(4):
            plt.subplot(4, 1, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
