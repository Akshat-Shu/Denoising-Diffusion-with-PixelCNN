import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, channels, device='cpu'):
        super().__init__()
        self.channels = channels
        self.device = device

        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)

        self.norm = nn.BatchNorm2d(channels)

        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2, device=device),
            nn.GELU(),
            nn.Linear(channels * 2, channels, device=device)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        x_flat = self.norm(x).view(b, c, -1).permute(0, 2, 1)

        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        attn_out = attn_out + x_flat

        attn_out = self.ff(attn_out)

        return attn_out.permute(0, 2, 1).view(b, c, h, w) + x




class DownSample(nn.Module):
    def __init__(self, channels, device='cpu', max_pool=True, use_attention=False):
        super().__init__()
        self.channels = channels
        self.device = device
        self.max_pool = max_pool

        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.use_attention = use_attention

        if use_attention:
            self.self_attention = SelfAttention(channels[1], device=device)

        if max_pool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if(x.shape[1] != self.channels[0]):
            raise ValueError(f"Input tensor must have {self.channels[0]} channels, but got {x.shape[1]} channels.")

        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        if self.use_attention:
            x = self.self_attention(x)

        x = self.conv2(x)

        x = self.relu(x)

        if self.max_pool:
            max_pool = self.maxpool(x)
            return max_pool, x
        
        return x