import numpy as np
import cv2


def ComputeHist(img):
    h, w = img.shape
    hist, bin_edge = np.histogram(img.reshape(1, w * h), bins=list(range(257)))
    return hist


def ComputeMinLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[i]
        if (sum >= (pnum * rate * 0.01)):
            return i


def ComputeMaxLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[255 - i]
        if (sum >= (pnum * rate * 0.01)):
            return 255 - i


def LinearMap(minlevel, maxlevel):
    if (minlevel >= maxlevel):
        return []
    else:
        newmap = np.zeros(256)
        for i in range(256):
            if (i < minlevel):
                newmap[i] = 0
            elif (i > maxlevel):
                newmap[i] = 255
            else:
                newmap[i] = (i - minlevel) / (maxlevel - minlevel) * 255
        return newmap


def CreateNewImg(img):
    h, w, d = img.shape
    newimg = np.zeros([h, w, d])
    for i in range(d):
        imgmin = np.min(img[:, :, i])
        imgmax = np.max(img[:, :, i])
        imghist = ComputeHist(img[:, :, i])
        minlevel = ComputeMinLevel(imghist, 8.3, h * w)
        maxlevel = ComputeMaxLevel(imghist, 2.2, h * w)
        newmap = LinearMap(minlevel, maxlevel)
        # print(minlevel, maxlevel)
        if (newmap.size == 0):
            continue
        for j in range(h):
            newimg[j, :, i] = newmap[img[j, :, i]]
    return newimg


if __name__ == '__main__':
    img = cv2.imread('ocr/1-5.png', 1)
    newimg = CreateNewImg(img)
    cv2.namedWindow('img', 0)
    cv2.imshow('img', img)
    cv2.namedWindow('newimg', 0)
    cv2.imshow('newimg', newimg / 255)
    cv2.waitKey(0)
