import torch

def kernel_filter(kernel_height, kernel_width, out_channels, in_channels, horizontal=False, first_layer=False, blinded=False, data_channels=1, device='cpu'):

    kernel_filter = torch.ones(out_channels, in_channels, kernel_height, kernel_width, device=device)

    mid_x = kernel_width // 2
    mid_y = kernel_height // 2

    if blinded or horizontal:
        kernel_filter[:, :, mid_y+1:, :] = 0.
        kernel_filter[:, :, mid_y, mid_x+1:] = 0.
    else:
        if first_layer:
            kernel_filter[:, :, mid_y:, :] = 0.
        else:
            kernel_filter[:, :, mid_y+1:, :] = 0.

    in_groups = in_channels // data_channels
    out_groups = out_channels // data_channels

    for o in range(data_channels):
        for i in range(o, data_channels):
            if i > o or (i == o and first_layer):
                kernel_filter[
                    o*out_groups:(o+1)*out_groups,
                    i*in_groups:(i+1)*in_groups,
                    mid_y, mid_x 
                ] = 0.

    return kernel_filter 