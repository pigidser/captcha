from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import io
import numpy as np
import cv2
import imutils

class_labels = ['2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K',
				'L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c',
				'd','e','f','g','h','j','k','m','n','p','q','r','s','t',
				'u','v','w','x','y','z']


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


def hsv_filter(image):

	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV )

	# начальный и конечный цвет фона
	h_min = np.array((115, 0, 0), np.uint8)
	h_max = np.array((116, 255, 255), np.uint8)
	
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

# Assign an instance of the FastAPI class to the variable "app".
# You will interact with your api using this instance.
app = FastAPI(title='Deploying a Captcha ML Model with FastAPI')

@app.on_event("startup")
def load():
	global model
	# load the pre-trained network
	print("[INFO] loading pre-trained network...")
	model = load_model("/app/minivggnet_1000_std.hdf5")

# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."

# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict") 
def prediction(file: UploadFile = File(...)):

	# 1. VALIDATE INPUT FILE
	filename = file.filename
	fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
	if not fileExtension:
		raise HTTPException(status_code=415, detail="Unsupported file provided.")
			
	# 2. TRANSFORM RAW IMAGE INTO CV2 image

	# Read image as a stream of bytes
	image_stream = io.BytesIO(file.file.read())

	# Start the stream from the beginning (position zero)
	image_stream.seek(0)

	# Write the stream of bytes into a numpy array
	file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

	# Decode the numpy array as an image
	image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

	if not (image.shape[0] == 160 and image.shape[1] == 520):
		raise HTTPException(status_code=416, detail="Image shape shoud be 520x160.")

    # 3. RUN OBJECT DETECTION MODEL
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	predictions = []
	part = gray.shape[1] // 4
	startY, endY = 20, 150

	for i in range(4):
		startX = part * i
		endX = startX + part
		roi = image[startY:endY, startX:endX]
		roi = hsv_filter(roi)
		# pre-process the ROI and classify it then classify it
		roi = preprocess(roi, 56, 56)
		roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
		pred = model.predict(roi).argmax(axis=1)[0]
		predictions.append(class_labels[pred])

	# return StreamingResponse("".join(predictions), media_type="application/json")
	return {"Prediction": "".join(predictions)}