import os
import shutil

imageDir = "dataset"
number_copies = 999

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
        print(file)
        new_file = os.path.splitext(file)[0]
        for j in range(dup):
            shutil.copy(
                os.path.join(imageDir, dir, file),
                os.path.join(imageDir, dir, new_file + "_" + str(j) + ".png"))
        
# walk = os.walk(imageDir)


# for root, dirs, files in os.walk(imageDir, topdown=False):
#     for dir in dirs:
#         print(f"-----------{dir}")
#         files = os.listdir(dir)
#         print(files)
    # if len(files) == 0:
    #     print("Folder")
    #     continue
    #     num_left = number_copies - len(files)
