import math
import time
import numpy as np
import matplotlib.pyplot as pl
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import torch.functional as F
import torchvision.models as models
import pandas as pd
import os

# Self defined
from CatsVsDogsData import CatsVsDogsDataset
from train_and_eval import train, eval

# HyperParameters
batch_size = 100
num_epochs = 10
num_classes = 2
learning_rate = 1e-3
in_channel = 3

# Transformer

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loading the data in
train_csv = pd.read_csv('train_csv.csv')

train_dataset = CatsVsDogsDataset(csv_file=train_csv, root_dir='cats_dogs_resized', transform=transform)

train_set, test_set = torch.utils.data.random_split(train_dataset, [20000, 5000])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Creating the model
model = models.googlenet(pretrained=True)

# Creating Loss and Optimizer

loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model=model, epochs=num_epochs, data_loader=train_loader, optimizer=optimizer, loss_func=loss_func,
      model_save_location=r'D:\PyCharm DL\CatsVsDogs\ModelSaveFolder')

eval(model=model, eval_dataloader=test_loader)

