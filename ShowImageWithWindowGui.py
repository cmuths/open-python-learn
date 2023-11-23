import numpy as np
import cv2
from matplotlib import  pyplot as plt

path = "./ocr/3-1.png"

im1 = cv2.imread(path)

fastnose = cv2.fastNlMeansDenoisingColored(im1, h=15, templateWindowSize=7, searchWindowSize=21)



upim=cv2.normalize(fastnose,dst=None,alpha=250,beta=10,norm_type=cv2.NORM_MINMAX)

b, g, r = cv2.split(upim)

split = r;

ret,thresh1=cv2.threshold(split,127,255,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(split,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3=cv2.threshold(split,127,255,cv2.THRESH_TRUNC)
ret,thresh4=cv2.threshold(split,127,255,cv2.THRESH_TOZERO)
ret,thresh5=cv2.threshold(split,127,255,cv2.THRESH_TOZERO_INV)



titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [im1, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
# cv2.imshow("demo",im2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

