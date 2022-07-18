# USAGE
# python basic_drawing.py

# import the necessary packages
import numpy as np
import cv2


for i in range(30):
	img = cv2.imread("0009.png") #1000x1000 pixel
	x1 = 0
	y1 = np.random.randint(5, img.shape[0] - 5)
	x2 = img.shape[1]
	y2 = np.random.randint(y1 - 30, y1 + 30)
	color = np.random.randint(0, high=256, size=(3,)).tolist()


	cv2.line(img, (x1, y1), (x2, y2), color, 2)
	cv2.imshow("Output", img)
	cv2.waitKey(0)