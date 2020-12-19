import os
import csv
from os import listdir
from os.path import isfile, join


def create_csv(dirname):
    # Get names of all images
    image_name_list = [f + ".jpg" for f in listdir(dirname) if isfile(join(dirname, f))]

    with open(r'D:\DeepLearning\CatVsDogsData\train.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for image_name in image_name_list:
            label = 0
            if(image_name.startswith("dog")):
                label = 1
            writer.writerow([image_name, label])

# Check if start has cat or dog and put label 1 or 0 with it.
# label =
# writer.writerow([image_name, label])

create_csv(r'D:\DeepLearning\CatVsDogsData\train')





