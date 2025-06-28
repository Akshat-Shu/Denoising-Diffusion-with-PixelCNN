import os
import torch

class Config:
    def __init__(self):
        self.model_name = "Cars_Diffusion"
        self.num_channels = 3
        self.hidden_channels = 3*64
        self.out_hidden_channels = 3*1024
        self.color_range = 256

        self.im_size = 64
        self.num_classes = 7
        self.n_layers = 12
        self.batch_size = 32

        self.epochs = 10
        self.num_workers = os.cpu_count()

        self.num_timesteps = 1000
        self.time_embedding_dim = 256
        self.embedding_dim = 1024

        self.lr = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")