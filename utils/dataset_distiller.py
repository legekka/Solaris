# This program is used to distill the dataset into a smaller dataset
# The dataset have sequences, each sequence is in a separate folder

import os
import pandas as pd

# get all csv files in the dataset folder
csv_files = []
for root, dirs, files in os.walk("dataset"):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

for file in csv_files:
    df = pd.read_csv(file)

    # print the number of rows in the csv file
    print(f"Number of rows in {file}: {len(df)}")

    # there's an images folder in the same folder as the csv file
    # we have to check if there's the same number of images as data in the csv file
    images_folder = os.path.join(os.path.dirname(file), "images")


    # We have to analyze the data:
    # - If the values don't change for a long time, we can remove the "boring" part by decreasing the datapoints, but it's very important to remove the corresponding images as well

    # check if the values are the same for a long time
    # at the minimum 50 frames must be the same

    i = 0
    while i < len(df) - 25:
        for j in range(i+1, len(df)):
            # go until the values are different
            if not (df.iloc[i] == df.iloc[j]).all():
                break
        # if the difference is more than 20 frames, remove the frames
        if j - i > 25:
            # we won't remove the datapoints yet, because we don't want to mess up the indexes, so we will just mark it by changing all values to -5 (this is a multidimensional datapoint, so each value will be -5)
            # but we will keep the first 5 frames, and the last 5 frames
            df.iloc[i+5:j-5] = -5


            # remove the corresponding images
            for k in range(i+5, j-5):
                #print(f"Removing {k+1}.png from {images_folder}")
                os.remove(os.path.join(images_folder, f"{k+1}.png"))
        i = j
        
    
    # remove the marked datapoints
    df = df[df.iloc[:, 0] != -5]

    print(f"Number of rows after removing boring parts from {file}: {len(df)}")

    # save into the same file
    df.to_csv(file, index=False)
    
    # we have to rename the images as well, since we removed some, and the counting is not continuous
    images = os.listdir(images_folder)

    # sort images by name
    images.sort(key=lambda x: int(x.split(".")[0]))
    for i in range(len(images)):
        os.rename(os.path.join(images_folder, images[i]), os.path.join(images_folder, f"{i+1}.png"))