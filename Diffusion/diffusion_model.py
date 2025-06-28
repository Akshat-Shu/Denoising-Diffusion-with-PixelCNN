import torch
import torch.nn as nn
from UNet.config import Config
from UNet.unet_model import UNet
from .time_embedding import TimeEmbedding
from .variance import VarianceSchedule
import tqdm

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = Config()
        # self.pixel_cnn = PixelCNN(self.cfg)
        # self.pixel_cnn.to(self.cfg.device)

        self.unet = UNet(device=self.cfg.device)

        self.time_embedding = TimeEmbedding(
            self.cfg.time_embedding_dim, 
            self.cfg.embedding_dim,
            self.cfg.num_timesteps,
            device=self.cfg.device
        )
        self.time_embedding.to(self.cfg.device)

        self.label_embedding = nn.Embedding(
            self.cfg.num_classes,
            self.cfg.embedding_dim
        )
        self.label_embedding.to(self.cfg.device)
        self.unet.to(self.cfg.device)

        self.var = VarianceSchedule(device=self.cfg.device)

        self.init_weights()

    def forward(self, x):
        images, labels, t = x
        images, labels, t = images.to(self.cfg.device), labels.to(self.cfg.device), t.to(self.cfg.device)
        time_emb = self.get_time_embedding(t)
        label_emb = self.get_label_embedding(labels)

        emb = time_emb + label_emb

        output = self.unet(images, label_emb, time_emb)

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
    
    def generate_images(self, n_samples=None, use_tqdm=False):
        self.unet.eval()
        with torch.inference_mode():
          noise_images = self.get_noise_image(n_samples)
          labels = torch.randint(
              0, self.cfg.num_classes, (n_samples,), device=self.cfg.device
          )
          label_emb = self.get_label_embedding(labels)
          new_image = noise_images.clone()
          
          rang = range(self.cfg.num_timesteps, 0, -1)
          if use_tqdm:
            rang = tqdm.tqdm(rang)

          for tt in rang:
              t = torch.tensor(tt, device=self.cfg.device).unsqueeze(0)
              
              time_emb = self.get_time_embedding(t)

              epsilon = self.unet(new_image, label_emb, time_emb)
              z = torch.randn_like(new_image, device=self.cfg.device) if t > 1 else torch.zeros_like(new_image, device=self.cfg.device)
              
              coeff = (1-self.var.alpha(t)) / \
              self.var.sqrt_one_minus_alpha_bar(t)

              new_image = (new_image - coeff * epsilon) / self.var.sqrt_alpha(t) + z * self.var.sqrt_beta(t)
              new_image = new_image.clamp(-1, 1)

        return new_image.cpu().detach().numpy(), labels.cpu().detach().numpy()
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)