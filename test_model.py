# USAGE
# python test_model.py --input input --model output/minivggnet_1000_std.hdf5 --size 10

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from pyimagesearch.utils import hsv_filter, preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from pyimagesearch import config

class_labels = ['2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K',
				'L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c',
				'd','e','f','g','h','j','k','m','n','p','q','r','s','t',
				'u','v','w','x','y','z']

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to input model")
ap.add_argument("-s", "--size", required=True, type=int,
	help="# images to test")
args = vars(ap.parse_args())

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# randomy sample a few of the input images
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(args["size"],),
	replace=False)

cv2.namedWindow( "result" ) # создаем главное окно

# loop over the image paths
for imagePath in imagePaths:
	# load the image and convert it to grayscale, then pad the image
	# to ensure digits caught only the border of the image are
	# retained
	image = cv2.imread(imagePath)

	assert image.shape[0] == 160
	assert image.shape[1] == 520

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	output = cv2.merge([gray] * 3)

	part = gray.shape[1] // 4
	startY = 20
	endY = 150
	w = 130
	h = 130

	predictions = []

	for i in range(4):
		startX = part * i
		endX = startX + part
		roi = image[startY:endY, startX:endX]

		roi = hsv_filter(roi, config.BACKGROUND_COLOR_H1, config.BACKGROUND_COLOR_H2)
		# cv2.imshow('result', roi)
		# cv2.waitKey()

		# pre-process the ROI and classify it then classify it
		roi = preprocess(roi, 56, 56)
		roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
		pred = model.predict(roi).argmax(axis=1)[0]
		predictions.append(class_labels[pred])

		# draw the prediction on the output image
		cv2.rectangle(output, (startX - 2, startY - 2),
			(startX + w + 4, startY + h + 4), (0, 255, 0), 1)
		cv2.putText(output, class_labels[pred], (startX + 10, startY - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


	# show the output image
	print("[INFO] captcha: {}".format("".join(predictions)))
	cv2.imshow("result", output)
	cv2.waitKey()