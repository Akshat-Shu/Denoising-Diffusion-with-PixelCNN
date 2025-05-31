from PIL import Image
import torchvision.transforms.functional as F
import random

class RandomPaddedCrop:
    def __init__(self, size, padding_mode='reflect'):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.padding_mode = padding_mode

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size

        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)

        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            img = F.pad(img, padding, padding_mode=self.padding_mode)

        i = random.randint(0, img.height - th)
        j = random.randint(0, img.width - tw)
        return F.crop(img, i, j, th, tw)
    

    def __reduce__(self):
        return (RandomPaddedCrop, (self.size,))