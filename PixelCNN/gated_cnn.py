import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernel_filter import kernel_filter
from collections import namedtuple


GatedCNNIO = namedtuple('GatedCNNIO', ['v', 'h', 'skip', 'label'])


class MaskedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_height=3, kernel_width=3, stride=1, padding='same', horizontal=False, first_layer=False, blinded=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride = stride
        self.padding = padding
        self.horizontal = horizontal
        self.first_layer = first_layer
        self.blinded = blinded

        self.W = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_height, kernel_width) * 0.1
        )
        self.bias_init()

        self.register_buffer('kernel_filter', kernel_filter(
            kernel_height, kernel_width, out_channels, in_channels, horizontal, first_layer, blinded
        ))

    def bias_init(self):
        self.b = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x):
        self.W.data *= self.kernel_filter
        return F.conv2d(
            x, self.W, self.b, 
            stride=self.stride, padding=self.padding
        )
    

class GatedCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_height=3, kernel_width=3, 
                 stride=1, padding='same', 
                 first_layer=False,
                 blinded=False, residual=False,
                 num_classes=7, embedding_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride = stride
        self.padding = padding
        self.first_layer = first_layer
        self.blinded = blinded
        self.residual = residual

        self.v_nxn = MaskedCNN(
            in_channels, 2*out_channels,
            kernel_height, kernel_width,
            stride, padding,
            horizontal=False, first_layer=first_layer, blinded=blinded
        )

        self.v_1x1 = MaskedCNN(
            2*out_channels, 2*out_channels,
            1, 1, stride, padding,
            horizontal=False, first_layer=first_layer, blinded=blinded
        )

        self.h_1xn = MaskedCNN(
            in_channels, 2*out_channels,
            1, kernel_width,
            stride, padding,
            horizontal=True, first_layer=first_layer, blinded=blinded
        )

        self.h_1x1 = MaskedCNN(
            out_channels, out_channels,
            1, 1, stride, padding,
            horizontal=True, first_layer=first_layer, blinded=blinded
        )

        if not self.first_layer:
            self.h_skip = MaskedCNN(
                out_channels, out_channels,
                1, 1, stride, padding,
                horizontal=True, first_layer=first_layer, blinded=blinded
            )

            self.embedding_proj = nn.Sequential(
                nn.GELU(),
                nn.Linear(embedding_dim, 2*out_channels),
            )


    def forward(self, x):
        v, h, skip, label, t = x
        v_conv = self.v_nxn(v)
        if h is None:
            h = v

        v_link = self.v_1x1(v_conv)

        h_conv = self.h_1xn(h)
        h_conv += v_link
        
        if not self.first_layer:
            combined_embedding = self.embedding_proj(label + t)
            h_conv += combined_embedding.unsqueeze(2).unsqueeze(3)

        h_gated = self.gated_unit(h_conv)
        v_gated = self.gated_unit(v_conv)

        if not self.first_layer:
            if skip is None:
                skip = self.h_skip(h_gated)
            else:
                skip += self.h_skip(h_gated)
        
        h_out = self.h_1x1(h_gated)
        
        
        if self.residual and not self.first_layer:
            h_out += h
        
        return v_gated, h_out, skip, label, t

    def gated_unit(self, T):
        T_f, T_g = torch.chunk(T, 2, dim=1)

        return torch.tanh(T_f) * torch.sigmoid(T_g)