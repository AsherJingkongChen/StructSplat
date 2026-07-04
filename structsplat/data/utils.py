
import torch
from PIL import Image
from torchvision import transforms as TF
from einops import rearrange
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torch.nn.functional import interpolate
from structsplat.utils.utils_2d import rescale_and_crop
from io import BytesIO
import numpy as np
from math import ceil

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__()) 

# def rescale_and_crop(img: Image.Image, shape: tuple):
#     w_in, h_in = img.size
#     w_out, h_out = shape

#     scale_factor = max(h_out / h_in, w_out / w_in)
#     h_scaled = round(h_in * scale_factor)
#     w_scaled = round(w_in * scale_factor)
#     assert h_scaled == h_out or w_scaled == w_out

#     img = img.resize((w_scaled, h_scaled), resample=Image.Resampling.BILINEAR)
#     left = (w_scaled - w_out) // 2
#     top = (h_scaled - h_out) // 2
#     img = img.crop((left, top, left + w_out, top + h_out))

#     return img

def load_and_preprocess_images(image_path_list, crop, new_size):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    new_h, new_w = new_size
    assert new_h <= new_w, "New height should be less than or equal to new width for the current implementation."
    assert new_h % 14 == 0 and new_w % 14 == 0, "New size should be divisible by 14 for the current implementation."

    images = []
    for image_path in image_path_list:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        img = TF.functional.to_tensor(img).float()
        if new_h < new_w and h > w:
            img = rearrange(img, "c h w -> c w h")
        img.unsqueeze_(0)
        if crop:
            img = rescale_and_crop(img, (new_h, new_w))
        else:
            img = interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True, antialias=True)        
        images.append(img)  # Remove the batch dimension
    images = torch.cat(images, dim=0)  # Concatenate along the batch dimension

    return images
