from torch import nn
import torch
from torch.utils.checkpoint import checkpoint

from structsplat.model.utils import ResidualConvUnit

class TextureEncoder(nn.Module):
    def __init__(self, features: int, with_activation_checkpoint: bool):
        super().__init__()

        self.img_res_conv_block = nn.Sequential(
            nn.Conv2d(3, features, kernel_size=3, stride=1, padding=1, bias=False),
            ResidualConvUnit(features, nn.ReLU)
        )
            
        self.feature_res_conv_block = ResidualConvUnit(features, nn.ReLU)
        self.forward = self._forward_with_ac if with_activation_checkpoint else self._forward_without_ac

    def _forward_with_ac(self, x, img) -> torch.Tensor:
        return checkpoint(self._forward_without_ac, x, img, use_reentrant=False)  # pyright: ignore[reportReturnType]

    def _forward_without_ac(self, x, img):   # pixel-wise addition
        output = self.img_res_conv_block(img)
        output = output + x
        output = self.feature_res_conv_block(output)
        return output


class GroupTextureEncoder(nn.Module):
    def __init__(self, features: int, with_activation_checkpoint: bool, groups: int = 5, padding_mode: str = 'replicate'):
        super().__init__()

        self.img_res_conv_block = nn.Sequential(
            nn.Conv2d(
                3, features*groups, kernel_size=3, stride=1, padding=1, bias=False, 
                padding_mode=padding_mode,
            ),
            ResidualConvUnit(
                features*groups, nn.ReLU, groups=groups, 
                padding_mode=padding_mode,
            )
        )
        
        self.output_res_conv_block = ResidualConvUnit(features*groups, nn.ReLU, groups=groups, 
            padding_mode=padding_mode,
        )
        self.forward = self._forward_with_ac if with_activation_checkpoint else self._forward_without_ac

    def _forward_with_ac(self, x, img) -> torch.Tensor:
        return checkpoint(self._forward_without_ac, x, img, use_reentrant=False)  # pyright: ignore[reportReturnType]

    def _forward_without_ac(self, x, img):   # pixel-wise addition
        output = self.img_res_conv_block(img)
        output = output + x
        output = self.output_res_conv_block(output)
        
        return output


        