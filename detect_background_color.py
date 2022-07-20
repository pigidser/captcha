# USAGE
# python detect_background_color.py

"""
Скрипт для получения фонового цвета каптчи. Двигать ползунок h1 и h2, чтобы
на черно-белой картинке получить белый фон и четкий контрастный символ.

"""
# import the necessary packages
import cv2
import numpy as np
import os
from pyimagesearch.utils import hsv_filter
from imutils import paths


if __name__ == '__main__':
    def nothing(*arg):
        pass

sample_folder = "symbol_samples"

# Create main in setting windows and move latter
cv2.namedWindow("color")
cv2.moveWindow("color", 800,300)
cv2.namedWindow("gray")
cv2.moveWindow("gray", 800,600)
cv2.namedWindow("settings")
cv2.moveWindow("settings", 40,30)

samples_list = [x for x in paths.list_images(sample_folder)]

# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('h1', 'settings', 115, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 116, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
cv2.createTrackbar('brightness', 'settings', 65, 255, nothing)
cv2.createTrackbar('contrast', 'settings', 170, 255, nothing)
cv2.createTrackbar('image number', 'settings', 0, len(samples_list) - 1, nothing)

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
    image_number = cv2.getTrackbarPos('image number', 'settings') - 1

    img = cv2.imread(samples_list[image_number])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

    # формируем начальный и конечный цвет фильтра
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    # накладываем фильтр на кадр в модели HSV
    filtered = cv2.inRange(hsv, h_min, h_max)

    img_filt = img.copy()
    img_filt[filtered > 0] = (0, 0, 0)

    gray_filt = hsv_filter(img.copy(), h1, h2)

    cv2.imshow('color', np.hstack([img, img_filt]))
    cv2.imshow('gray', gray_filt)

    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()