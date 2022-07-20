# USAGE
# python server.py

import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import nest_asyncio
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from utils import hsv_filter, preprocess

class_labels = ['2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K',
				'L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c',
				'd','e','f','g','h','j','k','m','n','p','q','r','s','t',
				'u','v','w','x','y','z']

model_filename = "../output/minivggnet_1000_std.hdf5"

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(model_filename)


# Assign an instance of the FastAPI class to the variable "app".
# You will interact with your api using this instance.
app = FastAPI(title='Deploying a Captcha ML Model with FastAPI')

# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."

# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict") 
def prediction(file: UploadFile = File(...)):

	# 1. Validate the input file
	filename = file.filename
	fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
	if not fileExtension:
		raise HTTPException(status_code=415, detail="Unsupported file provided.")
			
	# 2. Transform raw image into CV2 image

	# Read image as a stream of bytes
	image_stream = io.BytesIO(file.file.read())

	# Start the stream from the beginning (position zero)
	image_stream.seek(0)

	# Write the stream of bytes into a numpy array
	file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

	# Decode the numpy array as an image
	image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


    # 3. Run captcha recognition model

	if not (image.shape[0] == 160 and image.shape[1] == 520):
		raise HTTPException(status_code=416, detail="Image shape shoud be 520x160.")

	predictions = []

	# Our image size is 520x160 and it is consisted of 4 symbols
	# which we will recognize separately. We will pass in CNN the roi
	# with size 130x130.
	part = image.shape[1] // 4
	startY, endY = 20, 150

	for i in range(4):
		startX = part * i
		endX = startX + part
		# roi size will be 130x130
		roi = image[startY:endY, startX:endX]
		roi = hsv_filter(roi)

		# pre-process the ROI and classify it then classify it
		roi = preprocess(roi, 56, 56)
		roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
		pred = model.predict(roi).argmax(axis=1)[0]
		predictions.append(class_labels[pred])

	return {"prediction": "".join(predictions)}


# Allows the server to be run in this interactive environment
nest_asyncio.apply()

# Host depends on the setup you selected (docker or virtual env)
host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"

# Spin up the server!    
uvicorn.run(app, host=host, port=8000)