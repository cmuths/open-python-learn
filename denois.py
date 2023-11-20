import cv2

img = cv2.imread("ocr/1-2.png")
# img = cv2.fastNlMeansDenoisingColored(img,None,15,15,7,21)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret1, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# th5 = cv2.fastNlMeansDenoising(th5,None,15,7,21)
# th5 = cv2.medianBlur(th5,9)
img_th = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    1
)

cv2.imshow("show",th5)
cv2.waitKey(0)
cv2.destroyAllWindows()