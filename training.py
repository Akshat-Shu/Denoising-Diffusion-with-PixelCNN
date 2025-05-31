from Cars_Diffusion.Diffusion.variance import VarianceSchedule
from Diffusion.time_embedding import TimeEmbedding
import torch
from torch import nn
from torch.optim import AdamW
from Diffusion.diffusion_model import DiffusionModel


def train_model(model: DiffusionModel, test_loader, train_loader, epochs, device=None):
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            t = torch.randint(0, model.var.num_diffusion_timesteps, (images.size(0),), device=device)

            epsilon = torch.randn_like(images)
            noisy_images = model.var.