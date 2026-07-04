from einops import rearrange
from torch import nn
from structsplat.model.utils import ACTIVATION_DICT
from structsplat.model.group_dpt_head import get_group_dpt_head
from structsplat.model.texture_encoder import GroupTextureEncoder


class GroupGaussianPredictor(nn.Module):
    def __init__(self, features: int = 256, activations = None, padding_mode: str = 'replicate', gaussians_per_pixel = 1, with_tex_encoder = True, *args, **kwargs):
        super().__init__()
        self.channels = [1, 1, 3, 3, 4]
        self.gaussians_per_pixel = gaussians_per_pixel
        channels = [c * gaussians_per_pixel for c in self.channels]
        self.head_number = len(channels)
        assert isinstance(activations, list) and len(activations) == self.head_number, f"Activations must be a list of length {self.head_number}"
        self.activations = [ACTIVATION_DICT[act] for act in activations]
        self.conv_features = 64 if features >= 64 else 16
        self.gaussian_decoder = get_group_dpt_head(*args, features=features, groups=self.head_number, padding_mode=padding_mode, **kwargs)
    
        self.with_tex_encoder = with_tex_encoder
        if with_tex_encoder:
            self.tex_encoder = GroupTextureEncoder(features, True, groups=self.head_number, padding_mode=padding_mode) if with_tex_encoder else None
        
        self.proj = nn.Sequential(
            nn.Conv2d(features * self.head_number, self.conv_features * self.head_number, kernel_size=1, stride=1, padding=0, groups=self.head_number),
            nn.GELU(),
        )

        self.output_convs = nn.ModuleList([
            nn.Conv2d(self.conv_features, output_dim, kernel_size=3, stride=1, padding=1, 
                      padding_mode=padding_mode,
                      ) for output_dim in channels
        ])

    def forward(self, *args, **kwargs):
        img = kwargs["images"]
        B, S, _, _, _ = img.shape
        img = rearrange(img, "b s c h w -> (b s) c h w")
        x = self.gaussian_decoder.forward(*args, **kwargs)
        x = rearrange(x, "b s c h w -> (b s) c h w")
        if self.with_tex_encoder:
            x = self.tex_encoder(x, img)
        x = self.proj(x)
        x = rearrange(x, "b (g c) h w -> b g c h w", g=self.head_number, c=self.conv_features).unbind(1)
        features = []
        for i, xi, output_conv, act, c in zip(range(self.head_number), x, self.output_convs, self.activations, self.channels):
            xi = output_conv(xi)
            xi = rearrange(xi, "(b s) (g c) h w -> b s g c h w", b=B, s=S, g=self.gaussians_per_pixel, c=c)
            if i == self.head_number - 1:
                features.append(xi) # raw_rotation
            features.append(act(xi))
        
        return features
