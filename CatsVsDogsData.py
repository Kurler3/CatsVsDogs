import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io


# Dog images have labels 1 and cat images have labels 0
class CatsVsDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.__annotations__ = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.__annotations__)

    # Get item returns a specific image and it's label
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.__annotations__.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.__annotations__.iloc[index, 1]))

        if self.transform is not None:
            image = self.transform(image)

        return (image, y_label)