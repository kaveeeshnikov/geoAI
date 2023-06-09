import os
from PIL import Image
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import torchvision
from torchvision.datasets import ImageFolder
import transformers
from transformers import AutoImageProcessor, ResNetForImageClassification
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler
from sklearn.model_selection import train_test_split

model_name = ""
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
MODEL_SAVE_PATH = ""   # path to the folder where the model will be saved
DATASET_PATH = ""      # path to final dataframe with grids
PANORAMAS_PATH = ""    # path to panoramas
BATCH_SIZE = 512
LEARNING_RATE = 3e-5
N_EPOCH = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv(DATASET_PATH)
NUM_OF_CLASSES = max(df['index_grid'])
df["image"] = PANORAMAS_PATH + df["image"] + ".jpg"
df = df.rename(columns={"image": "image_path"})
df = df.dropna()
df["index_grid"] = df["index_grid"].astype(int)

def get_train_test_split():
    X, y = df["image_path"], df["index_grid"]
    train_images, test_images, train_labels, test_labels = train_test_split(
        X, y, test_size=0.15, random_state=333
    )
    train_images.reset_index(drop=True, inplace=True)
    test_images.reset_index(drop=True, inplace=True)
    test_labels.reset_index(drop=True, inplace=True)
    train_labels.reset_index(drop=True, inplace=True)
    return train_images, train_labels, test_images, test_labels

class GeoGuessrDataset(Dataset):
    def __init__(self, images_path, targets, processor=processor):
        self.images = images_path
        self.labels = targets
        self.input_processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.input_processor(images=image)
        target = self.labels[index]

        return image, target

def unfreeze_layers(model):
    """
    Unfreeze the weights of all stages except the first.

    """
    for i, param in enumerate(model.named_parameters()):
        if param[0].startswith(('classifier', 'resnet.encoder.stages.3', 'resnet.encoder.stages.2')):
            param[1].requires_grad = True
        else:
            param[1].requires_grad = False

unfreeze_layers(model)
# change output of the last fc layer
model.classifier[1] = nn.Linear(in_features=2048, out_features=NUM_OF_CLASSES) 
model = model.to(device)

train_images, train_targets, test_images, test_targets = get_train_test_split()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
optimizer.zero_grad()

test_dataloader = DataLoader(
    dataset=GeoGuessrDataset(test_images, test_targets),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)
train_dataloader = DataLoader(
    dataset=GeoGuessrDataset(train_images, train_targets),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)

def show_batch_images(images, labels):
    plt.figure(figsize=(20, 15))
    for n in range(len(images)):
        ax = plt.subplot(8, 8, n + 1)
        plt.imshow(images[n].permute(1, 2, 0))
        plt.title(int(labels[n]))
        plt.axis("off")

# training loop
best_epoch = 0
best_loss = 0

for epoch in range(N_EPOCH):
    epoch_loss = 0

    for i, data in enumerate(train_dataloader):
        images, labels = data
        images = images["pixel_values"]
        images = torch.stack(images)
        images = images.squeeze(0)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    epoch_loss /= len(train_dataloader)
    print(f"epoch: {epoch + 1}, training loss: {epoch_loss}")
    """
    test loss
    
    """
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            images, labels = data
            images = images["pixel_values"]
            images = torch.stack(images)
            images = images.squeeze(0)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            epoch_loss += loss.item()
        epoch_loss /= len(test_dataloader)
    print(f"epoch: {epoch + 1}, test loss: {epoch_loss}")

    #saves model if current epoch_loss better
    if epoch_loss < best_loss or best_loss == 0:
        best_loss = epoch_loss
        best_epoch = epoch + 1
        print(f"save best model at {best_loss} with epoch {best_epoch}")
        torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}{model_name}.pt")

    # early stop if the loss does not stop grow
    if epoch - 10 > best_epoch:
        print(
            f"early stop at {epoch+1} with best epoch {best_epoch} and test similarity {best_loss}."
        )
        break
