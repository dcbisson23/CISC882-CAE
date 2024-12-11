import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image
import pandas as pd
import numpy as np

img_transform = transforms.Compose([
    # Each image must be dimensions 1024 x 1024
    transforms.Resize((1024, 1024)),
    transforms.Grayscale(),
    transforms.ConvertImageDtype(torch.float)
])

# These need to be DataSet objects.
root_dir = os.path.dirname(__file__)

# We need to make files for train/test subsets!
# Alternatively, we could make a mapper that can reference each image in its original folder.
entry_dir = "data\\Data_Entry_2017.csv"
data_entries = pd.read_csv(os.path.join(root_dir, entry_dir))
# A list of the first file in each image folder.
file_splits = ["00000001_000.png", "00001336_000.png", "00003923_014.png", "00006585_007.png",
               "00009232_004.png", "00011558_008.png", "00013774_026.png", "00016051_010.png",
               "00018387_035.png", "00020945_050.png", "00024718_000.png", "00028173_003.png"]
# The list of image folder directories. The next loop adds the root and other bits.
img_dirs = ["images_001", "images_002", "images_003", "images_004",
            "images_005", "images_006", "images_007", "images_008",
            "images_009", "images_010", "images_011", "images_012"]
for i in range(len(img_dirs)):
    data_dir = os.path.join(root_dir, "data")
    img_dirs[i] = os.path.join(data_dir, img_dirs[i])
    img_dirs[i] = os.path.join(img_dirs[i], "images")
img_dirs = np.array(img_dirs)

class ChestXRayDataset(Dataset):
    """Custom dataset of chest XRay images."""

    def __init__(self, transform=img_transform, subset=None, labels=None, store_as_tensor=True):
        """
        :param transform: Any necessary transforms for the input images.
        :param subset: A list of .
        Only images
        """
        self.transform = transform
        self.dir_map = []

        self.tensor = None
        self.classes = None

        if store_as_tensor:
            self.tensor = torch.tensor([])

        if subset is None:
            subset = data_entries

        for img_id in subset["Image Index"]:
            # In case the image isn't an image. Don't think this will come up though.
            if not img_id.endswith(".png"):
                break

            # We need to find which image directory the target is in.
            dir_num = 0
            curr_dir = list(os.listdir(img_dirs[0]))
            while img_id >= file_splits[dir_num + 1]:
                dir_num += 1
                if dir_num + 1 >= len(file_splits):
                    break
            img_path = os.path.join(img_dirs[dir_num], img_id)

            # This is the part where we apply the transforms to the image and save the tensor.
            img = read_image(str(img_path))
            self.dir_map.append(img)
            if self.tensor is not None:
                if self.transform:
                    img = self.transform(img[:3, :, :])
                    img = torch.unsqueeze(img, 0)
                self.tensor = torch.cat([self.tensor, img], dim=0)
        # We need these as np arrays for performance reasons (Python lists cause memory leaks with DataLoaders).
        self.dir_map = np.array(self.dir_map)

        if labels is not None:
            self.classes = []
            for img_labels in subset["Finding Labels"]:
                img_labels = img_labels.split(separator="|")
                img_classes = []
                for label in labels:
                    # If we have the "No Finding" label, it means we're doing a one-class classifier.
                    # Thus, we want only one class, which represents whether a finding exists or not.
                    if label == "No Finding":
                        img_classes = [0 if label in img_labels else 1]
                        break
                    # This is the most likely outcome, if testing for a particular pathology (or pathologies).
                    img_classes.append(1 if label in img_labels else 0)
                self.classes.append(img_classes)
            # We need this as a np array for performance reasons (same as above).
            self.classes = np.array(self.classes)

    def __len__(self):
        return len(self.dir_map)

    def __getitem__(self, idx):
        # This segment determines which file path contains the indexed image.
        if self.tensor is not None:
            img = self.tensor[idx, :]

        else:
            img_path = self.dir_map[idx]
            img = read_image(str(img_path))
            if self.transform:
                img = self.transform(img[:3, :, :])
                img = torch.unsqueeze(img, 0)

        # We also want to include the classes if using a supervised model.
        if self.classes is None:
            return img
        else:
            return img, self.classes[idx]

    def is_labelled(self) -> bool:
        return True if self.classes is not None else False
