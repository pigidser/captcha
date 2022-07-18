import cv2
import numpy as np
import os
from utils import *


if __name__ == '__main__':
    def nothing(*arg):
        pass

cv2.namedWindow( "result" ) # создаем главное окно

image_file = "../0202.png"
image_folder = "preprocessing"

file = os.path.join(os.getcwd(), image_folder, image_file)

if not os.path.isfile(file):
    raise Exception(f"File not found!")

img = cv2.imread(file)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )

kernelSize = (3, 3)
# apply an "average" blur to the image using the current kernel
blurred = cv2.GaussianBlur(gray, kernelSize, 0)

wide = cv2.Canny(blurred, 10, 200)
# mid = cv2.Canny(blurred, 30, 150)
# tight = cv2.Canny(blurred, 240, 250)

eroded = cv2.erode(gray.copy(), None, iterations=1)
dilated = cv2.dilate(gray.copy(), None, iterations=2)

kernelSize = (7, 7)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

kernelSize = (7, 7)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

kernelSize = (7, 7)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

# construct a rectangular kernel (13x5) and apply a blackhat
# operation which enables us to find dark regions on a light
# background
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

# construct a rectangular kernel (13x5) and apply a blackhat
# operation which enables us to find dark regions on a light
# background
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)


# cv2.imshow('result', np.hstack([gray, wide, eroded, dilated, opening, closing, gradient, blackhat, tophat]))
cv2.imshow('result', np.hstack([gray, blurred, wide]))

ch = cv2.waitKey()

cv2.destroyAllWindows()