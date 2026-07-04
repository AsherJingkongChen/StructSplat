from einops import rearrange
from transformers import DINOv3ConvNextBackbone
from torchvision.transforms import Normalize
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

def get_semantic_encoder(sem_encoder_type, path, img_size):
    if sem_encoder_type is None:
        return None
    elif sem_encoder_type == "dinov3_convnext_large":
        return DinoV3ConvnextLarge(path, img_size)
    else:
        raise NotImplementedError(f"Semantic dec type {sem_encoder} not implemented.")

class DinoV3ConvnextLarge(nn.Module):
    def __init__(self, path, img_size):
        super().__init__()
        self.extractor = DINOv3ConvNextBackbone.from_pretrained(path)

        self.dino_normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.new_size = (ceil(img_size[0] / 32) * 32, ceil(img_size[1] / 32) * 32)
       
    def forward(self, img):
        B, S, _, _, _ = img.shape
        img = rearrange(img, "b s c h w -> (b s) c h w")
        img = F.interpolate(img, size=self.new_size, mode='bilinear', align_corners=True, antialias=True)
        img = self.dino_normalize(img)
        output = self.extractor(img, output_hidden_states=True).hidden_states
        output = [rearrange(feat, "(b s) c h w -> b s c h w", b=B, s=S) for feat in output]
        return output