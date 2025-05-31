import torch

class VarianceSchedule:
    def __init__(self, beta_start=0.0001, beta_end=0.02, timesteps=1000):
        self.num_diffusion_timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def variance(self):
        def get(t):
            if t <= 0 or t > self.num_diffusion_timesteps:
                raise ValueError("t must be in the range [0, num_diffusion_timesteps - 1]")
            return self.betas[t-1]
        return get

    @property
    def alpha_bar(self):
        def get(t):
            if t <= 0 or t > self.num_diffusion_timesteps:
                raise ValueError("t must be in the range [0, num_diffusion_timesteps - 1]")
            return self.alphas_cumprod[t-1]
        return get

    @property
    def alpha(self):
        def get(t):
            if t <= 0 or t > self.num_diffusion_timesteps:
                raise ValueError("t must be in the range [0, num_diffusion_timesteps - 1]")
            return self.alphas[t-1]
        return get