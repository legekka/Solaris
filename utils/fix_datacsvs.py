# load every csv file in the dataset folder (recursively)
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
    # there's an images folder in the same folder as the csv file
    # we have to check if there's the same number of images as data in the csv file
    images_folder = os.path.join(os.path.dirname(file), "images")
    images = os.listdir(images_folder)
    # likely there will be more data than images, so we need to remove the extra data
    if len(df) > len(images):
        print(f"Removing {len(df) - len(images)} rows from {file}")
        df = df.iloc[:len(images)]
        df.to_csv(file, index=False)
    # if there are more images than data, we need to remove the extra images
    elif len(images) > len(df):
        for i in range(len(df), len(images)):
            os.remove(os.path.join(images_folder, f"{i+1}.png"))

