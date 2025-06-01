import torch
import torch.nn as nn
from PixelCNN.config import Config
from PixelCNN.model import PixelCNN
from .time_embedding import TimeEmbedding
from .variance import VarianceSchedule


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = Config()
        self.pixel_cnn = PixelCNN(self.cfg)

        self.time_embedding = TimeEmbedding(
            self.cfg.time_embedding_dim, 
            self.cfg.embedding_dim,
            self.cfg.num_timesteps
        )

        self.label_embedding = nn.Embedding(
            self.cfg.num_classes,
            self.cfg.embedding_dim
        )

        self.var = VarianceSchedule()

    def forward(self, x):
        images, labels, t = x
        time_emb = self.get_time_embedding(t)
        label_emb = self.get_label_embedding(labels)

        emb = time_emb + label_emb

        output = self.pixel_cnn((images, label_emb, time_emb))

        return output, emb
    
    def get_time_embedding(self, t):
        return self.time_embedding(t)
    
    def get_label_embedding(self, labels):
        return self.label_embedding(labels)
    
    def get_noise_image(self, n_samples=None):
        if n_samples is None:
            n_samples = self.cfg.batch_size
        return torch.randn(
            (n_samples, self.cfg.num_channels, self.cfg.im_size, self.cfg.im_size),
            device=self.cfg.device
        )
    
    def generate_images(self, n_samples=None):
        noise_images = self.get_noise_image(n_samples)
        labels = torch.randint(
            0, self.cfg.num_classes, (n_samples,), device=self.cfg.device
        )
        label_emb = self.get_label_embedding(labels)
        new_image = noise_images.clone()
        
        for t in range(self.cfg.num_timesteps, 0, -1):
            time_emb = self.get_time_embedding(t)
            label_emb = self.get_label_embedding(labels)
            
            epsilon = self.pixel_cnn((new_image, label_emb, time_emb))
            z = torch.randn_like(new_image) if t > 1 else torch.zeros_like(new_image)
            coeff = (1-self.var.alpha(t)) / self.var.sqrt_one_minus_alpha_bar(t)

            new_image = (new_image - coeff * epsilon) / self.var.sqrt_alpha_bar(t) + z * self.var.variance(t)

        return new_image