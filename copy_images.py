# USAGE
# python copy_images.py --dataset dataset1 --copies 19

# import the necessary packages
import os
import argparse
import shutil

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset")
ap.add_argument("-n", "--copies", required=True,
	help="number of copies to create")

args = vars(ap.parse_args())

imageDir = args["dataset"]
number_copies = int(args["copies"])

for dir in os.listdir(imageDir):
    if not os.path.isdir(os.path.join(imageDir,dir)):
        continue
    print(f"--- {dir} ---")
    files = os.listdir(os.path.join(imageDir, dir))
    if len(files) == 0:
        print(f"Folder {dir} is empty!")
        continue
    dup = number_copies // len(files)
    for i, file in enumerate(files):
        new_file = os.path.splitext(file)[0]
        for j in range(dup):
            shutil.copy(
                os.path.join(imageDir, dir, file),
                os.path.join(imageDir, dir, new_file + "_" + str(j) + ".png"))
        