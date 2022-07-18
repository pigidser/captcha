import cv2
import numpy as np

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def apply_transform(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

    # формируем начальный и конечный цвет фильтра, яркость и контрастность
    h_min = np.array((70, 40, 70), np.uint8)
    h_max = np.array((255, 255, 255), np.uint8)
    # brightness, contrast = 17 - 127, 210 - 127

    # накладываем фильтр на кадр в модели HSV
    filtered = cv2.inRange(hsv, h_min, h_max)
    # subtracted = cv2.subtract(filtered, gray)
    # negative = cv2.bitwise_not(subtracted)
    # res = apply_brightness_contrast(negative, brightness, contrast)

    return filtered