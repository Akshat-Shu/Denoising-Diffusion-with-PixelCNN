import torch
import torch.nn as nn
from UNet.contatenate import CenterConcat

class UpSample(nn.Module):
    def __init__(self, channels, device='cpu'):
        super().__init__()
        self.channels = channels
        self.device = device

        self.up_conv = nn.ConvTranspose2d(
            channels[0], channels[1], kernel_size=2, stride=2
        )
        self.bn_1 = nn.BatchNorm2d(channels[1])

        self.conv1 = nn.Conv2d(2*channels[1], channels[2], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1)

        self.relu = nn.ReLU()

        self.concat = CenterConcat(device=device)

    def forward(self, x):
        if not isinstance(x, tuple):
            raise ValueError("Input must be a tuple containing the tensor and the skip connection tensor.")
        
        if len(x) != 2:
            raise ValueError("Input tuple must contain exactly two tensors.")
        
        x, skip = x

        if x.shape[1] != self.channels[0]:
            raise ValueError(f"Input tensor must have {self.channels[0]} channels, but got {x.shape[1]} channels.")
        
        if skip.shape[1] != self.channels[1]:
            raise ValueError(f"Skip tensor must have {self.channels[1]} channels, but got {skip.shape[1]} channels.")
        
        x = self.up_conv(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.concat(skip, x)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        return x