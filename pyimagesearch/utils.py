import cv2
import numpy as np
import imutils

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


def draw_line(image):
	# Draw a random line
	x1, y1 = 0, np.random.randint(5, image.shape[0] - 5)
	x2, y2 = image.shape[1], np.random.randint(y1 - 30, y1 + 30)
	color = np.random.randint(0, high=256, size=(3,)).tolist()
	cv2.line(image, (x1, y1), (x2, y2), color, 2)
	
	return image


def hsv_filter(image, h1, h2):

	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV )

	# начальный и конечный цвет фона
	h_min = np.array((h1, 0, 0), np.uint8)
	h_max = np.array((h2, 255, 255), np.uint8)
	
	# накладываем фильтр на изображение в модели HSV и меняем цвет фона на черный
	filtered = cv2.inRange(hsv, h_min, h_max)
	image[filtered > 0] = (0, 0, 0)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
	negative = cv2.bitwise_not(gray)

	# Contrast and brightness
	brightness, contrast = 15 - 127, 255 - 127
	adjusted = apply_brightness_contrast(negative, brightness, contrast)

    # apply an "average" blur to the image using the current kernel
	kernelSize = (3, 3)
	blurred = cv2.GaussianBlur(adjusted, kernelSize, 1)

	return blurred


def preprocess(image, width, height):

	# grab the dimensions of the image, then initialize
	# the padding values
	(h, w) = image.shape[:2]

	# if the width is greater than the height then resize along
	# the width
	if w > h:
		image = imutils.resize(image, width=width)

	# otherwise, the height is greater than the width so resize
	# along the height
	else:
		image = imutils.resize(image, height=height)

	# determine the padding values for the width and height to
	# obtain the target dimensions
	padW = int((width - image.shape[1]) / 2.0)
	padH = int((height - image.shape[0]) / 2.0)

	# pad the image then apply one more resizing to handle any
	# rounding issues
	image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
		cv2.BORDER_REPLICATE)
	image = cv2.resize(image, (width, height))

	# return the pre-processed image
	return image