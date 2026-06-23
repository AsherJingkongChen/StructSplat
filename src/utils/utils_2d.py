import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.nn.functional import interpolate


# def rescale(
#     image: Float[Tensor, "B 3 h_in w_in"],
#     shape: tuple[int, int],
# ) -> Float[Tensor, "B 3 h_out w_out"]:
#     new_h, new_w = shape
#     image = interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=True, antialias=True)
#     return image


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    shape: tuple[int, int],
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    return images


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    shape: tuple[int, int],
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    # assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = interpolate(images, size=(h_scaled, w_scaled), mode='bilinear', align_corners=True, antialias=True)
    images = images.reshape(*batch, c, h_scaled, w_scaled)
    images = center_crop(images, shape)
    return images
