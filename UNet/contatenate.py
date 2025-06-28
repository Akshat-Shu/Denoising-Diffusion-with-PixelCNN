import torch
import torch.nn as nn

class CenterConcat(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def forward(self, x1, x2):
        if(x1.shape[0] != x2.shape[0]):
            raise ValueError("Input tensors must have the same batch size.")
        
        if(x1.shape[1] != x2.shape[1]):
            raise ValueError("Input tensors must have the same number of channels.")
        
        # # print("x1 shape:", x1.shape, "x2 shape:", x2.shape)
        # x1 = x1.to(device=self.device)
        # x2 = x2.to(device=self.device)

        # center = x1.shape[2] // 2
        # output_dim = x2.shape[2]
        # if output_dim % 2 != 0:
        #     raise ValueError("The output dimension must be even for center concatenation.")
        
        # half_dim = output_dim // 2
        # # 32 - 2x = 28 -> x = 2 3...30 = 2...29, center = 16 half = 14 center-half, center+half-1
        # x1_cropped = x1[:, :, center-half_dim:center+half_dim, center-half_dim:center+half_dim]
        x1_cropped = x1

        return torch.cat((x1_cropped, x2), dim=1)
