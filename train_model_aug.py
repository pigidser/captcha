# USAGE
# python train_model_aug.py --dataset dataset --model output/lenet.hdf5
# python train_model_aug.py --dataset dataset --model output/minivggnet.hdf5
# python train_model_aug.py -d dataset -m output/minivggnet_200_rand.hdf5 -e 50 -s standard

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from pyimagesearch.learning_rate_schedulers import StepDecay
from pyimagesearch.learning_rate_schedulers import PolynomialDecay

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf

from pyimagesearch import config
from pyimagesearch.nn.conv import MiniVGGNet
from pyimagesearch.utils.captchahelper import draw_line, hsv_filter, preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# define training hyperparameters
BATCH_SIZE = 64
# epochs = 40

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-e", "--epochs", type=int, default=50,
	help="# of epochs to train for")
ap.add_argument("-s", "--schedule", type=str, default="",
	help="learning rate schedule method")

args = vars(ap.parse_args())

# store the number of epochs to train for in a convenience variable,
# then initialize the list of callbacks and learning rate scheduler
# to be used
epochs = args["epochs"]
callbacks = []
schedule = None

# check to see if step-based learning rate decay should be used
if args["schedule"] == "step":
	print("[INFO] using 'step-based' learning rate decay...")
	schedule = StepDecay(initAlpha=config.INIT_LR, factor=0.25, dropEvery=15)

# check to see if linear learning rate decay should should be used
elif args["schedule"] == "linear":
	print("[INFO] using 'linear' learning rate decay...")
	schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=config.INIT_LR, power=1)

# check to see if a polynomial learning rate decay should be used
elif args["schedule"] == "poly":
	print("[INFO] using 'polynomial' learning rate decay...")
	schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=config.INIT_LR, power=5)

# if the learning rate schedule is not empty, add it to the list of
# callbacks
if schedule is not None:
	callbacks = [LearningRateScheduler(schedule)]

# initialize an early stopping callback to prevent the model from
# overfitting
callbacks.append(EarlyStopping(
	monitor="val_loss",
	patience=config.EARLY_STOPPING_PATIENCE,
	restore_best_weights=True)
)

# initialize the decay for the optimizer
decay = 0.0

# if we are using Keras' "standard" decay, then we need to set the
# decay parameter
if args["schedule"] == "standard":
	print("[INFO] using 'keras standard' learning rate decay...")
	decay = config.INIT_LR / epochs

# otherwise, no learning rate schedule is being used
elif schedule is None:
	print("[INFO] no learning rate schedule being used")


# initialize the data and labels
data = []
labels = []

# cv2.namedWindow( "result" ) # создаем главное окно

# loop over the input images
print("Prepare our dataset...")
for imagePath in paths.list_images(args["dataset"]):
	# load the image, pre-process it, and store it in the data list
	# print(imagePath)
	image = cv2.imread(imagePath)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	image = draw_line(image)
	image = hsv_filter(image)
	# cv2.imshow('result', image)
	# cv2.waitKey()

	image = preprocess(image, 56, 56)
	image = img_to_array(image)
	
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainLabels, testLabels) = train_test_split(data,
	labels, test_size=0.20, random_state=42)

# convert the labels from integers to vectors
lb = LabelEncoder().fit(trainLabels)
trainLabels = lb.transform(trainLabels)
testLabels = lb.transform(testLabels)

print(lb.classes_)
# exit()


# initialize our sequential data augmentation pipeline for training
trainAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255),
	# preprocessing.RandomFlip("horizontal_and_vertical"),
	preprocessing.RandomTranslation(
		height_factor=(-0.1, 0.2),
		width_factor=(-0.1, 0.2),
		fill_mode='nearest',
		interpolation='bilinear',
		seed=42)
	# preprocessing.RandomRotation(0.3)
])

# initialize a second data augmentation pipeline for testing (this
# one will only do pixel intensity rescaling
testAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255)
])

# # scale the raw pixel intensities to the range [0, 1]
# data = np.array(data, dtype="float") / 255.0
# labels = np.array(labels)


# prepare the training data pipeline (notice how the augmentation
# layers have been mapped)
trainDS = tf.data.Dataset.from_tensor_slices((trainX, trainLabels))
trainDS = (
	trainDS
	.shuffle(BATCH_SIZE * 100)
	.batch(BATCH_SIZE)
	.map(lambda x, y: (trainAug(x), y),
		 num_parallel_calls=tf.data.AUTOTUNE)
	.prefetch(tf.data.AUTOTUNE)
)


# create our testing data pipeline (notice this time that we are
# *not* apply data augmentation)
testDS = tf.data.Dataset.from_tensor_slices((testX, testLabels))
testDS = (
	testDS
	.batch(BATCH_SIZE)
	.map(lambda x, y: (testAug(x), y),
		num_parallel_calls=tf.data.AUTOTUNE)
	.prefetch(tf.data.AUTOTUNE)
)


# initialize the model
print("[INFO] compiling model...")
model = MiniVGGNet.build(width=56, height=56, depth=1, classes=55)
# initialize our optimizer and model, then compile it
opt = SGD(learning_rate=config.INIT_LR, momentum=0.9, decay=decay)
# opt = SGD(lr=0.04)
# opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# model.compile(loss="categorical_crossentropy", optimizer=opt,
# 	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
# epochs = 20
# H = model.fit(trainX, trainY,  validation_data=(testX, testY),
# 	batch_size=32, epochs=epochs, verbose=1)

H = model.fit(
	trainDS,
	validation_data=testDS,
	epochs=epochs,
	callbacks=[callbacks])


# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(testDS)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testDS, batch_size=32)
# print(classification_report(testLabels.argmax(axis=1),
# 	predictions.argmax(axis=1), target_names=lb.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# plt.savefig(args["plot"])

# # plot the training + testing loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, epochs), H.history["accuracy"], label="acc")
# plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
plt.show()