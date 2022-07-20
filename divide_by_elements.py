# USAGE
# python divide_by_elements.py --input input --dataset dataset1

import cv2
import os
import argparse
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to folder with initial captcha images")
ap.add_argument("-d", "--dataset", required=True,
	help="path to folder dataset")

args = vars(ap.parse_args())

rawImageDir = args["input"]
imageDir = args["dataset"]

if not os.path.isdir(rawImageDir):
    raise Exception(f"Folder {rawImageDir} not found!")
    
if not os.path.exists(imageDir):
    os.mkdir(imageDir)

counter = 1

for rawImagePath in paths.list_images(os.path.join(os.getcwd(), rawImageDir)):
    image = cv2.imread(rawImagePath)

    assert image.shape[0] == 160
    assert image.shape[1] == 520

    part = image.shape[1] // 4
    startY, endY = 20, 150

    for i in range(4):
        startX = part * i
        endX = startX + part
        roi = image[startY:endY, startX:endX]
        imagePath = os.path.join(imageDir, f"{str(counter).zfill(4)}.png")
        cv2.imwrite(imagePath, roi)
        counter += 1