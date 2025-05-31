## Denoising Diffusion using Gated PixelCNN

PyTorch implementation of the paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) that was trained to generate images of cars using a dataset I found on kaggle: [Car Images Dataset](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset)

The decoder was implemented using a Gated PixelCNN model with 12 layers as described in the paper [Conditional Image Generation with PixelCNN Decoders](http://arxiv.org/abs/1606.05328)