import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint_sequential
import torch.fft as fft
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from preprocessing import ChestXRayDataset
import os
import pandas as pd
from PIL import Image
from torchsummary import summary
import torchvision.models.resnet as resnet
# Foundational code sourced from
# https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/

from cae import LeakyCAE
from resnetcae import ResNetCAE
from traintest import ModelTrainer


if __name__ == "__main__":
    # Initialize the autoencoder
    # model = Autoencoder()
    model = ResNetCAE(resnet.Bottleneck, [3, 4, 6, 3])

    root_dir = os.path.dirname(__file__)

    entry_dir = "data\\Data_Entry_2017.csv"
    data_entries = pd.read_csv(os.path.join(root_dir, entry_dir))

    normal_entries = data_entries.loc[data_entries["Finding Labels"] == "No Finding"]
    print(len(normal_entries))
    # Partition the normal entries for unsupervised training.
    normal_train = normal_entries.sample(frac=0.0001, replace=False, random_state=42).sort_index()
    print(len(normal_train))
    # Remember the rest of the set.
    normal_entries = normal_entries.drop(normal_train.index)

    train_dataset = ChestXRayDataset(subset=normal_train)

    trainer = ModelTrainer(model, train_dataset)
    trainer.train(save_dir='models/ResNetCAE', batch_size=1, num_epochs=100)


