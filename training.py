from Diffusion.variance import VarianceSchedule
from Diffusion.time_embedding import TimeEmbedding
import torch
from torch import nn
from torch.optim import AdamW
from Diffusion.diffusion_model import DiffusionModel



def train_step(model: DiffusionModel, batch, optimizer, criterion, device = None):
    if device is None:
        device = model.cfg.device
    model.train()
    images, labels = batch

    images, labels = images.to(device), labels.to(device)
    t = torch.randint(0, model.var.num_diffusion_timesteps, (images.size(0),), device=device)
    t_reshaped = t.view(-1, 1, 1, 1)

    epsilon = torch.randn_like(images).detach()
    noisy_images = model.var.sqrt_alpha_bar(t_reshaped) * images + \
        model.var.sqrt_one_minus_alpha_bar(t_reshaped) * epsilon
    
    optimizer.zero_grad()
    predicted_noise, _ = model((noisy_images, labels, t))
    
    loss = criterion(predicted_noise, epsilon)
    loss.backward()

    optimizer.step()

    return loss.item()

def test_step(model: DiffusionModel, batch, criterion, device = None):
    if device is None:
        device = model.cfg.device
    model.eval()
    images, labels = batch

    images, labels = images.to(device), labels.to(device)
    t = torch.randint(0, model.var.num_diffusion_timesteps, (images.size(0),), device=device)

    t_reshaped = t.view(-1, 1, 1, 1)

    epsilon = torch.randn_like(images)
    noisy_images = model.var.sqrt_alpha_bar(t_reshaped) * images + \
        model.var.sqrt_one_minus_alpha_bar(t_reshaped) * epsilon
    
    with torch.no_grad():
        predicted_noise, _ = model((noisy_images, labels, t))
    
    loss = criterion(predicted_noise, epsilon)

    return loss.item()