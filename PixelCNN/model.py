import torch
import torch.nn as nn
from .gated_cnn import GatedCNNIO, GatedCNNBlock, MaskedCNN
from .config import Config
from torchvision import transforms

class PixelCNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.cfg = config
        
        self.first_layer = GatedCNNBlock(
            in_channels=config.num_channels,
            out_channels=config.hidden_channels,
            kernel_height=7, kernel_width=7,
            first_layer=True, residual=False,
            blinded=False, embedding_dim=self.cfg.embedding_dim,
        )

        self.hidden_layers = nn.ModuleList([
            GatedCNNBlock(
                in_channels=config.hidden_channels,
                out_channels=config.hidden_channels,
                kernel_height=3, kernel_width=3,
                first_layer=False, residual=True,
                blinded=False, num_classes=config.num_classes,
                embedding_dim=self.cfg.embedding_dim
            ) for _ in range(config.n_layers)
        ])

        self.embedding_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.hidden_channels),
        )

        self.fl1 = MaskedCNN(
            config.hidden_channels, config.out_hidden_channels,
            kernel_height=1, kernel_width=1,
            stride=1, padding='same',
            horizontal=True, first_layer=False, blinded=False
        )

        self.fl2 = MaskedCNN(
            config.out_hidden_channels, config.num_channels * config.color_range,
            kernel_height=1, kernel_width=1,
            stride=1, padding='same',
            horizontal=True, first_layer=False, blinded=False
        )

        self.relu = nn.ReLU()

        self.normalize = transforms.Normalize(
            mean=[0.5*config.color_range, 0.5*config.color_range, 0.5*config.color_range],
            std=[0.5*config.color_range, 0.5*config.color_range, 0.5*config.color_range]
        )

        self.training = True

    def forward(self, x):
        images, labels, t = x
        skip = torch.tensor(0) # for torchinfo
        io = (images, images, skip, labels, t)

        io = self.first_layer(io)
        v, h, skip, label, t = io

        skip = torch.zeros_like(h)
        io = (v, h, skip, labels, t)
        for layer in self.hidden_layers:
            io = layer(io)
        
        v, h, skip, labels, t = io

        label_embedding = self.embedding_proj(labels + t).unsqueeze(2).unsqueeze(3)
        
        out = skip + label_embedding
        out = self.relu(out)
        out = self.fl1(out)
        out = self.relu(out)
        out = self.fl2(out)

        out = out.view(out.shape[0], self.cfg.num_channels, self.cfg.color_range, out.shape[2], out.shape[3])
        out = out.permute(0, 1, 3, 4, 2)
        if self.training:
            out = torch.nn.functional.gumbel_softmax(out, tau=1.0, hard=False, dim=-1)
            out = torch.sum(out * torch.arange(self.cfg.color_range, device=out.device), dim=-1)
        else:
            out = torch.argmax(out, dim=-1)

        return self.normalize(out)
    
    def toggle_train(self):
        self.training = not self.training
        if self.training:
            self.normalize = transforms.Normalize(
                mean=[0.5*self.cfg.color_range, 0.5*self.cfg.color_range, 0.5*self.cfg.color_range],
                std=[0.5*self.cfg.color_range, 0.5*self.cfg.color_range, 0.5*self.cfg.color_range]
            )
        else:
            self.normalize = transforms.ToTensor()