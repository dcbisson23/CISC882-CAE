
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import os

from preprocessing import ChestXRayDataset


class ModelTrainer:

    def __init__(self, model: nn.Module) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        self.criterion = None
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.use_cudnn_benchmark = True
        self.cur_epoch = 0

    def load_pretrained(self, pt_path) -> None:
        """Load pretrained weights into the model from a .pt file. Only works if .pt file is from same model type."""
        pt_model = torch.load(pt_path)
        self.model.load_state_dict(pt_model['model_state_dict'])
        self.optimizer.load_state_dict(pt_model['optimizer_state_dict'])
        self.cur_epoch = pt_model['epoch']

    def train(self, dataset: ChestXRayDataset, save_dir, num_epochs=100, batch_size=8, lr=0.001) -> None:
        """For a given dataset, train the model on the dataset. Save snapshots of each epoch to the given directory."""
        self.model.to(self.device)

        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=4)
        if lr:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        torch.backends.cudnn.benchmark = self.use_cudnn_benchmark

        supervised = dataset.is_labelled()
        if supervised:
            self.criterion = nn.CrossEntropyLoss
        else:
            self.criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            epoch_loss = 0
            iters = 0
            for data in data_loader:
                # https://wandb.ai/wandb_fc/tips/reports/How-To-Implement-Gradient-Accumulation-in-PyTorch--VmlldzoyMjMwOTk5
                # torch.cuda.empty_cache()
                classes = None
                if supervised:
                    img = data[0]
                    classes = data[1]
                    classes.to(self.device, non_blocking=True)
                else:
                    img = data
                img = img.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                output = self.model(img)
                if supervised:
                    loss = self.criterion(output, classes)
                else:
                    loss = self.criterion(output, img)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                iters += 1
            epoch_loss = epoch_loss / iters

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))

            # Save the model
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': epoch_loss},
                       os.path.join(save_dir, 'epoch-{}.pt'.format(epoch)))
            self.cur_epoch += 1


class ModelTester:
    """
    The ModelTester class is used to handle a given pre-trained model for testing and validation.
    This may need to be changed, depending on our methods of testing/validation (ex. k-fold CV...)

    """
    def __init__(self, model: nn.Module) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        self.criterion = None

    def load_weights(self, pt_path) -> None:
        """Load pretrained weights into the model from a .pt file. Only works if .pt file is from same model type."""
        pt_model = torch.load(pt_path, weights_only=True)
        self.model.load_state_dict(pt_model['model_state_dict'])

    def test(self, dataset: ChestXRayDataset) -> []:
        """
        For a given set of data, get the model's output and evaluate the reconstruction loss.
        This will return an array of loss values for each sample.
        Note adjustments may need to be made if any other metrics are to be implemented (ex. k-fold CV)
        """
        self.model.to(self.device)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 pin_memory=True,
                                 num_workers=1)

        supervised = dataset.is_labelled()
        if supervised:
            self.criterion = nn.CrossEntropyLoss
        else:
            self.criterion = nn.MSELoss()

        loss = []
        for data in data_loader:
            # https://wandb.ai/wandb_fc/tips/reports/How-To-Implement-Gradient-Accumulation-in-PyTorch--VmlldzoyMjMwOTk5
            # torch.cuda.empty_cache()
            classes = None
            if supervised:
                img = data[0]
                classes = data[1]
                classes.to(self.device, non_blocking=True)
            else:
                img = data

            img = img.to(self.device, non_blocking=True)
            output = self.model(img)

            if supervised:
                data_loss = self.criterion(output, classes)
            else:
                data_loss = self.criterion(output, img)

            loss.append(data_loss.cpu().detach().numpy())
        return loss

    def sample(self, dataset, save_dir) -> None:
        """Create and save comparison images for a given set of input image samples."""
        self.model.to(self.device)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 pin_memory=True,
                                 num_workers=1)

        supervised = dataset.is_labelled()

        to_pil = transforms.ToPILImage(mode="L")
        sample_num = 0
        for data in data_loader:
            if supervised:
                img = data[0]
            else:
                img = data
            img = img.to(self.device, non_blocking=True)
            output = self.model(img)
            base = to_pil(torch.reshape(data, (1024, 1024)))
            reconstruction = to_pil(torch.reshape(output, (1024, 1024)))
            comparison = Image.fromarray(np.hstack((np.array(base), np.array(reconstruction))))
            comparison.save(os.path.join(save_dir, 'sample-{}.png'.format(sample_num)))
            sample_num += 1
