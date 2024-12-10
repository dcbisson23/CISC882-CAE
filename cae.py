import torch
import torch.nn as nn



class LeakyCAE(nn.Module):
    def __init__(self):
        super(LeakyCAE, self).__init__()
        # Each image starts as a 1024 x 1024 x (1?) bitmap.
        # I'll keep track of dimensions in comments as we move through the encoding/decoding layers.
        self.encoder = nn.Sequential(
            # 1024 x 1024 x 1
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # 1024 x 1024 x 48
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            # 512 x 512 x 48
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            # 512 x 512 x 64
            nn.LeakyReLU(0.01),
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            # 256 x 256 x 64
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            # 256 x 256 x 82
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #  128 x 128 x 82
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # 128 x 128 x 96
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            # 64 x 64 x 64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # 64 x 64 x 128
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01)
            # 32 x 32 x 128
            # THIS is the size of the latent space.
        )
        self.decoder = nn.Sequential(
            # THIS is the size of the latent space.
            nn.ConvTranspose2d(128, 128,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # 128 x 128 x 96
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               output_padding=0),
            nn.LeakyReLU(0.01),
            # 64 x 64 x 96
            nn.ConvTranspose2d(64, 64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # 128 x 128 x 96
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               output_padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               output_padding=0),
            # 128 x 128 x 82
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # 256 x 256 x 82
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 24,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               output_padding=0),
            # 256 x 256 x 64
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(24, 24,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # 512 x 512 x 64
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(24, 16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               output_padding=0),
            # 512 x 512 x 48
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # 1024 x 1024 x 48
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 1,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               output_padding=0),
            # 1024 x 1024 x 1
            nn.Sigmoid()  # No dimension change, just standardizes range of each pixel value to [0, 1] in greyscale.
            # 1024 x 1024 x 1
            # THIS is the output size (should be equal to input size).
        )

    def forward(self, x):
        # x = self.encoder()
        # x = self.decoder()
        x = torch.utils.checkpoint.checkpoint_sequential(self.encoder, 4, x, use_reentrant=True)
        x = torch.utils.checkpoint.checkpoint_sequential(self.decoder, 4, x, use_reentrant=True)
        return x