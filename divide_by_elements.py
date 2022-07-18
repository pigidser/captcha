import cv2
import os
from imutils import paths

rawImageDir = "raw"
imageDir = "dataset"
# image = cv2.imread(imagePath)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# for imagePath in paths.list_images("dataset/raw"):
#     print(imagePath)

counter = 1

for rawImagePath in paths.list_images(os.path.join(os.getcwd(), rawImageDir)):
    image = cv2.imread(rawImagePath)

    assert image.shape[0] == 160
    assert image.shape[1] == 520

    part = image.shape[1] // 4
    startY = 20
    endY = 150 # image.shape[0]
    for i in range(4):
        startX = part * i
        endX = startX + part
        roi = image[startY:endY, startX:endX]
        imagePath = os.path.join(imageDir, f"{str(counter).zfill(4)}.png")
        cv2.imwrite(imagePath, roi)
        print(startY, endY, startX, endX)
        counter += 1
