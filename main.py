import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import preprocessing
import testwithleaky
import torchvision.models.resnet as resnet
import resnetcae
import os
import pandas as pd
from PIL import Image
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import ChestXRayDataset
from traintest import ModelTrainer, ModelTester
from resnetcae import ResNetCAE
from cae import LeakyCAE

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

# Create an untrained model instance.
# Note: These are the only two we have implemented currently. Choose as needed.
model = ResNetCAE(resnet.Bottleneck, [3, 4, 6, 3])
# model = LeakyCAE()

train_dataset = ChestXRayDataset(subset=normal_train)

trainer = ModelTrainer(model)
trainer.train(dataset=train_dataset, save_dir='models/ResNetCAE', batch_size=1, num_epochs=100)
good_entries = data_entries.loc[data_entries["Finding Labels"] == "No Finding"]
bad_entries = data_entries.drop(good_entries.index)

good_entries = good_entries.sample(n=1, replace=True, random_state=42)
bad_entries = bad_entries.sample(n=1, replace=True, random_state=42)

good_data = preprocessing.ChestXRayDataset(subset=good_entries)
bad_data = preprocessing.ChestXRayDataset(subset=bad_entries)


tester = ModelTester(model)
# Make sure this is a relative path from the project root.
# Point it where the pretrained model weights are.
tester.load_weights("models/ResNetCAE/epoch1.pt")

good_loss = tester.test(good_data)
bad_loss = tester.test(bad_data)

# EVALUATION STARTS HERE
# Note: I haven't touched this in a while. I'd recommend messing with how we evaluate the model.
print(np.mean(good_loss), np.var(good_loss))
print(np.mean(bad_loss), np.var(bad_loss))
if abs(np.var(good_loss) - np.var(bad_loss)) > 10:
    summary = stats.ttest_ind(good_loss, bad_loss, equal_var=False)

else:
    summary = stats.ttest_ind(good_loss, bad_loss, equal_var=True)
    print(summary)

all_loss = [good_loss, bad_loss]
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(all_loss)
plt.show()