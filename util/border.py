import cv2 as cv
import numpy as np


class Border:
    def canny(self, img, thresh):
        edges = cv.Canny(img, thresh, 255)
        # 使用膨胀和腐蚀来改善边缘形状
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        dilated = cv.dilate(edges, kernel, iterations=2)
        # dilated = cv.medianBlur(dilated,5)
        eroded = cv.erode(dilated, kernel, iterations=1)
        mask = np.ones((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
        cv.floodFill(edges, mask, (0, 0), (255, 255, 255), (30, 30, 30), (30, 30, 30), cv.FLOODFILL_FIXED_RANGE)
        return eroded
