import torch
import torch.nn as nn
from UNet.downsample import DownSample
from UNet.upsample import UpSample
from UNet.config import Config

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, device='cpu'):
        super().__init__()
        self.cfg = Config()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.down_1 = DownSample(
            [in_channels, 64, 64], device, max_pool=True
        )

        self.down_2 = DownSample(
            [64, 128, 128], device, max_pool=True
        )

        self.down_3 = DownSample(
            [128, 256, 256], device, max_pool=True, use_attention=True
        )

        self.down_4 = DownSample(
            [256, 512, 512], device, max_pool=True, use_attention=True
        )

        self.down_5 = DownSample(
            [512, 1024, 1024], device, max_pool=False, use_attention=True
        )

        self.up_1 = UpSample(
            [1024, 512, 512, 512], device
        )

        self.up_2 = UpSample(
            [512, 256, 256, 256], device
        )

        self.up_3 = UpSample(
            [256, 128, 128, 128], device
        )

        self.up_4 = UpSample(
            [128, 64, 64, 64], device
        )

        self.embedding_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.cfg.embedding_dim, 1024)
        )

        self.skip_1_proj = nn.Sequential(nn.GELU(), nn.Linear(self.cfg.embedding_dim, 64))
        self.skip_2_proj = nn.Sequential(nn.GELU(), nn.Linear(self.cfg.embedding_dim, 128))
        self.skip_3_proj = nn.Sequential(nn.GELU(), nn.Linear(self.cfg.embedding_dim, 256))
        self.skip_4_proj = nn.Sequential(nn.GELU(), nn.Linear(self.cfg.embedding_dim, 512))

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x, label_embedding, time_embedding):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Input tensor must have {self.in_channels} channels, but got {x.shape[1]} channels.")

        down_1, skip_1 = self.down_1(x)
        down_2, skip_2 = self.down_2(down_1)
        down_3, skip_3 = self.down_3(down_2)
        down_4, skip_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)

        embedding = self.embedding_proj(label_embedding + time_embedding)
        embedding = embedding.unsqueeze(2).unsqueeze(3)
        down_5 = down_5 + embedding

        skip_1 = skip_1 + self.skip_1_proj(label_embedding + time_embedding).unsqueeze(2).unsqueeze(3)
        skip_2 = skip_2 + self.skip_2_proj(label_embedding + time_embedding).unsqueeze(2).unsqueeze(3)
        skip_3 = skip_3 + self.skip_3_proj(label_embedding + time_embedding).unsqueeze(2).unsqueeze(3)
        skip_4 = skip_4 + self.skip_4_proj(label_embedding + time_embedding).unsqueeze(2).unsqueeze(3)


        up_1 = self.up_1((down_5, skip_4))
        up_2 = self.up_2((up_1, skip_3))
        up_3 = self.up_3((up_2, skip_2))
        up_4 = self.up_4((up_3, skip_1))

        return self.final_conv(up_4)