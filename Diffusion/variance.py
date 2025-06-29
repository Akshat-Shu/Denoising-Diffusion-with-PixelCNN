import torch

class VarianceSchedule:
    def __init__(self, beta_start=0.0001, beta_end=0.02, timesteps=1000, device='cpu'):
        self.num_diffusion_timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.sigmas = torch.sqrt(self.betas)
        self.alphas = 1.0 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def variance(self):
        def get(t):
            if isinstance(t, torch.Tensor):
                if (torch.any(t <= 0) or torch.any(t > self.num_diffusion_timesteps)):
                    raise ValueError("t must be a tensor with values in the range [1, num_diffusion_timesteps]")
            elif t <= 0 or t > self.num_diffusion_timesteps:
                raise ValueError("t must be in the range [0, num_diffusion_timesteps - 1]")
            return self.betas[t-1]
        return get

    @property
    def alpha_bar(self):
        def get(t):
            if isinstance(t, torch.Tensor):
                if (torch.any(t <= 0) or torch.any(t > self.num_diffusion_timesteps)):
                    raise ValueError("t must be a tensor with values in the range [1, num_diffusion_timesteps]")
            elif t <= 0 or t > self.num_diffusion_timesteps:
                raise ValueError("t must be in the range [0, num_diffusion_timesteps - 1]")
            return self.alphas_cumprod[t-1]
        return get

    @property
    def alpha(self):
        def get(t):
            if isinstance(t, torch.Tensor):
                if (torch.any(t <= 0) or torch.any(t > self.num_diffusion_timesteps)):
                    raise ValueError("t must be a tensor with values in the range [1, num_diffusion_timesteps]")
            elif t <= 0 or t > self.num_diffusion_timesteps:
                raise ValueError("t must be in the range [0, num_diffusion_timesteps - 1]")
            return self.alphas[t-1]
        return get
    
    @property
    def sqrt_alpha_bar(self):
        def get(t):
            if isinstance(t, torch.Tensor):
                if (torch.any(t <= 0) or torch.any(t > self.num_diffusion_timesteps)):
                    raise ValueError("t must be a tensor with values in the range [1, num_diffusion_timesteps]")
            elif t <= 0 or t > self.num_diffusion_timesteps:
                raise ValueError("t must be in the range [0, num_diffusion_timesteps - 1]")
            return self.sqrt_alphas_cumprod[t-1]
        return get
    
    @property
    def sqrt_one_minus_alpha_bar(self):
        def get(t):
            if isinstance(t, torch.Tensor):
                if (torch.any(t <= 0) or torch.any(t > self.num_diffusion_timesteps)):
                    raise ValueError("t must be a tensor with values in the range [1, num_diffusion_timesteps]")
            elif t <= 0 or t > self.num_diffusion_timesteps:
                raise ValueError("t must be in the range [0, num_diffusion_timesteps - 1]")
            return self.sqrt_one_minus_alphas_cumprod[t-1]
        return get
    
    @property
    def sqrt_alpha(self):
        def get(t):
            if isinstance(t, torch.Tensor):
                if (torch.any(t <= 0) or torch.any(t > self.num_diffusion_timesteps)):
                    raise ValueError("t must be a tensor with values in the range [1, num_diffusion_timesteps]")
            elif t <= 0 or t > self.num_diffusion_timesteps:
                raise ValueError("t must be in the range [0, num_diffusion_timesteps - 1]")
            return torch.sqrt(self.alphas[t-1])
        return get
    

    @property
    def sqrt_beta(self):
        def get(t):
            if isinstance(t, torch.Tensor):
                if (torch.any(t <= 0) or torch.any(t > self.num_diffusion_timesteps)):
                    raise ValueError("t must be a tensor with values in the range [1, num_diffusion_timesteps]")
            elif t <= 0 or t > self.num_diffusion_timesteps:
                raise ValueError("t must be in the range [0, num_diffusion_timesteps - 1]")
            return torch.sqrt(self.betas[t-1])
        return get 