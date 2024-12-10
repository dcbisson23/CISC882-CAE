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


class ChestXRayDataset(Dataset):
    """Custom dataset of chest XRay images."""

    def __init__(self, transform=img_transform, subset=None):
        """
        :param transform: Any necessary transforms for the input images.
        :param subset: A list of .
        Only images
        """
        self.img_dir = img_dirs
        self.transform = transform
        self.dir_map = []
        self.dir_splits = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tensor = torch.tensor([])
        self.tensor.to(self.device)

        self.subsets = None
        if subset is not None:
            self.subsets = [[] for _ in range(len(img_dirs))]
            for img in subset["Image Index"]:
                dir_num = 0
                while img >= file_splits[dir_num + 1]:
                    dir_num += 1
                    if dir_num + 1 >= len(file_splits):
                        break
                self.subsets[dir_num].append(os.listdir(img_dirs[dir_num]).index(img))
        # This section lets us address images by an index in the directory, rather than by their file name.
        # It will be useful when looping through the directory.

        # Now, prune the map for non-PNG files and any file not indexed in subsets.
        for i in range(len(self.img_dir)):
            curr_dir = list(os.listdir(self.img_dir[i]))
            self.dir_splits.append(0)
            for j in range(len(curr_dir) - 1):
                file = curr_dir[j]
                if file.endswith(".png"):
                    if self.subsets:
                        if j in self.subsets[i]:
                            self.dir_map.append(file)
                            self.dir_splits[i] += 1
                            img_path = os.path.join(self.img_dir[i], file)
                            image = read_image(str(img_path))
                            image.to(self.device)
                            # Apply any necessary transforms.
                            if self.transform:
                                image = self.transform(image[:3, :, :])
                                image = torch.unsqueeze(image, 0)
                            self.tensor = torch.cat([self.tensor, image], dim=0)
                    else:
                        self.dir_map.append(file)
                        self.dir_splits[i] += 1
                        img_path = os.path.join(self.img_dir[i], file)
                        image = read_image(str(img_path))
                        # Apply any necessary transforms.
                        if self.transform:
                            image = self.transform(image[:3, :, :])
                        self.tensor = torch.cat([self.tensor, image], dim=0)

        self.img_dir = np.array(self.img_dir)
        self.dir_map = np.array(self.dir_map)
        self.dir_splits = np.array(self.dir_splits)
        self.tensor.to(self.device)

    def __len__(self):
        return len(self.dir_map)

    def __getitem__(self, idx):
        # This segment determines which file path contains the indexed image.
        # dir_num = 0
        # accum = self.dir_splits[0]
        # while idx >= accum and dir_num < len(self.img_dir):
        #     dir_num += 1
        #     accum += self.dir_splits[dir_num]
        # # Get the image from the file path.
        # img_path = os.path.join(self.img_dir[dir_num], self.dir_map[idx])
        # image = read_image(str(img_path))
        # # Apply any necessary transforms.
        # if self.transform:
        #     image = self.transform(image[:3, :, :])
        # image.to(self.device)
        image = self.tensor[idx]
        return image
