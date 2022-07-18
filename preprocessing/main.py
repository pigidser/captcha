import cv2
import numpy as np
import os
from utils import *


if __name__ == '__main__':
    def nothing(*arg):
        pass

cv2.namedWindow( "result" ) # создаем главное окно
cv2.namedWindow( "settings" ) # создаем окно настроек

# image_file = "51-page-8.png"
image_file = "../0029.png"
image_folder = "preprocessing"

file = os.path.join(os.getcwd(), image_folder, image_file)

if not os.path.isfile(file):
    raise Exception(f"File not found!")

img = cv2.imread(file)

# создаем 6 бегунков для настройки начального и конечного цвета фильтра
# # Variant 1
# cv2.createTrackbar('h1', 'settings', 90, 255, nothing)
# cv2.createTrackbar('s1', 'settings', 15, 255, nothing)
# cv2.createTrackbar('v1', 'settings', 85, 255, nothing)
# cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
# cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
# cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
# cv2.createTrackbar('brightness', 'settings', 5, 255, nothing)
# cv2.createTrackbar('contrast', 'settings', 255, 255, nothing)
# Variant 2
cv2.createTrackbar('h1', 'settings', 115, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 125, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
cv2.createTrackbar('brightness', 'settings', 65, 255, nothing)
cv2.createTrackbar('contrast', 'settings', 170, 255, nothing)

while True:
 
    # считываем значения бегунков
    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')
    
    brightness = cv2.getTrackbarPos('brightness', 'settings') - 127
    contrast = cv2.getTrackbarPos('contrast', 'settings') - 127

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

    # формируем начальный и конечный цвет фильтра
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    # накладываем фильтр на кадр в модели HSV
    filtered = cv2.inRange(hsv, h_min, h_max)

    img_filt = img.copy()
    img_filt[filtered>0] = (0,0,0)
    gray_filt = cv2.cvtColor(img_filt, cv2.COLOR_BGR2GRAY )

    filtered_negative = cv2.bitwise_not(filtered)

    subtracted = cv2.subtract(filtered, gray)

    negative = cv2.bitwise_not(subtracted)

    # adjusted = negative
    adjusted = apply_brightness_contrast(negative, brightness, contrast)

    kernelSize = (3, 3)
    # apply an "average" blur to the image using the current kernel
    blurred = cv2.GaussianBlur(gray, kernelSize, 0)
    wide = cv2.Canny(blurred, 10, 200)

    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # cv2.imshow('result', np.hstack([gray, gray_filt, equalized, filtered, filtered_negative, subtracted, negative, adjusted, wide]))
    cv2.imshow('result', np.hstack([img, img_filt]))

    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()