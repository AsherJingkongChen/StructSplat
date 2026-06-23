from typing import List, Tuple, Union
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from vggt.heads.utils import create_uv_grid, position_grid_to_embed
from model.utils import ResidualConvUnit


def get_group_dpt_head(*args, sem_encoder_type=None, **kwargs):
    if sem_encoder_type is None:
        return GroupDPTHead(*args, **kwargs)
    elif sem_encoder_type == "dinov3_convnext_large":
        return GroupDPTHeadForDinoV3CL(*args, **kwargs)
    else:
        raise ValueError(f"Semantic encoder type {sem_encoder_type} not implemented.")


class GroupDPTHead(nn.Module):
    """
    Base Group DPT Head implementations.
    """
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        down_ratio: int = 1,
        groups: int = 5,
        padding_mode: str = 'zeros',
        *args,
        **kwargs,
    ) -> None:
        super(GroupDPTHead, self).__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.groups = groups
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc * groups, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0] * groups, out_channels=out_channels[0] * groups, kernel_size=4, stride=4, padding=0, groups=groups
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1] * groups, out_channels=out_channels[1] * groups, kernel_size=2, stride=2, padding=0, groups=groups
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3] * groups, out_channels=out_channels[3] * groups, kernel_size=3, stride=2, padding=1, groups=groups,
                    padding_mode=padding_mode
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False, groups=groups,
                                     padding_mode=padding_mode
                                     )

        # Attach additional modules to scratch.
        self.scratch.refinenet1 = _make_fusion_block(features, groups=groups, shallow_ffb=True, padding_mode=padding_mode)
        self.scratch.refinenet2 = _make_fusion_block(features, groups=groups, padding_mode=padding_mode)
        self.scratch.refinenet3 = _make_fusion_block(features, groups=groups, padding_mode=padding_mode)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False, groups=groups, padding_mode=padding_mode)
        
        self.out_channels = out_channels

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        S = images.shape[1]

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            chunk_output = self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
            )
            all_preds.append(chunk_output)

        # Concatenate results along the sequence dimension
        return torch.cat(all_preds, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            
            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1]) # B*S, H'*W', C

            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)) # B*S, C, H', W'
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        return out.view(B, S, *out.shape[1:])


    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1]).to(x.dtype)
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        return out


# dinov3 convnext version
class GroupDPTHeadForDinoV3CL(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        down_ratio: int = 1,
        groups: int = 5,

        sem_channels: List[int] = [192, 384, 768, 1536],
        padding_mode: str = 'zeros',
    ) -> None:
        super(GroupDPTHeadForDinoV3CL, self).__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.groups = groups
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc * groups, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0] * groups, out_channels=out_channels[0] * groups, kernel_size=4, stride=4, padding=0, groups=groups
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1] * groups, out_channels=out_channels[1] * groups, kernel_size=2, stride=2, padding=0, groups=groups
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3] * groups, out_channels=out_channels[3] * groups, kernel_size=3, stride=2, padding=1, groups=groups,
                    padding_mode=padding_mode
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False, groups=groups,
                                     padding_mode=padding_mode
                                     )

        # Attach additional modules to scratch.
        self.scratch.refinenet1 = _make_fusion_block(features, groups=groups, shallow_ffb=True, padding_mode=padding_mode)
        self.scratch.refinenet2 = _make_fusion_block(features, groups=groups, padding_mode=padding_mode)
        self.scratch.refinenet3 = _make_fusion_block(features, groups=groups, padding_mode=padding_mode)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False, groups=groups, padding_mode=padding_mode)

        self.sem_projects = nn.ModuleList([
            nn.Conv2d(in_channels=dc, out_channels=oc * groups, kernel_size=1, stride=1, padding=0)
            for dc, oc in zip(sem_channels, out_channels)
        ])
        
        self.projects2 = nn.ModuleList([
            nn.Conv2d(in_channels=2 * oc * groups, out_channels=oc * groups, kernel_size=1, stride=1, padding=0, groups=groups)
            for oc in out_channels
        ])

        self.sem_norms = nn.ModuleList([
            nn.LayerNorm(dc) for dc in sem_channels
        ])
        
        self.out_channels = out_channels


    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
        sem_feature_list: List[torch.Tensor] | None = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        S = images.shape[1]

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx, sem_feature_list=sem_feature_list)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            chunk_output = self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx, sem_feature_list=sem_feature_list
            )
            all_preds.append(chunk_output)

        # Concatenate results along the sequence dimension
        return torch.cat(all_preds, dim=1)


    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,

        sem_feature_list: List[torch.Tensor] | None = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            
            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)) # B*S, C, H', W'
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            _, _, x_h, x_w = x.shape
            x_sem = sem_feature_list[dpt_idx]
            if frames_start_idx is not None and frames_end_idx is not None:
                x_sem = x_sem[:, frames_start_idx:frames_end_idx]
            x_sem = rearrange(x_sem, "b s c h w -> (b s) h w c")
            x_sem = self.sem_norms[dpt_idx](x_sem)
            x_sem = rearrange(x_sem, "b h w c -> b c h w")
            x_sem = self.sem_projects[dpt_idx](x_sem)
            x_sem = F.interpolate(x_sem, size=(x_h, x_w), mode='bilinear', align_corners=True, antialias=True)
            x_sem = rearrange(x_sem, "b (g c) h w -> b g c h w", g=self.groups, c=self.out_channels[dpt_idx])
            x = rearrange(x, "b (g c) h w -> b g c h w", g=self.groups, c=self.out_channels[dpt_idx])
            x = torch.cat([x, x_sem], dim=2)
            x = rearrange(x, "b g c h w -> b (g c) h w")
            x = self.projects2[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        return out.view(B, S, *out.shape[1:])


    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1]).to(x.dtype)
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        return out



################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1, shallow_ffb: bool = False, padding_mode: str = "zeros") -> nn.Module:
    return FeatureFusionBlock(
        features * groups,
        nn.ReLU,
        deconv=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
        shallow_ffb=shallow_ffb,
        padding_mode=padding_mode
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False,
                  padding_mode: str = "zeros"
                  ) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0]*groups, out_shape1*groups, kernel_size=3, stride=1, padding=1, bias=False, groups=groups,
        padding_mode=padding_mode
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1]*groups, out_shape2*groups, kernel_size=3, stride=1, padding=1, bias=False, groups=groups,
        padding_mode=padding_mode
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2]*groups, out_shape3*groups, kernel_size=3, stride=1, padding=1, bias=False, groups=groups,
        padding_mode=padding_mode
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3]*groups, out_shape4*groups, kernel_size=3, stride=1, padding=1, bias=False, groups=groups,
            padding_mode=padding_mode
        )
    return scratch

class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
        shallow_ffb=False,
        padding_mode="zeros"
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        if shallow_ffb:
            self.out_conv = nn.Conv2d(
                features, out_features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups, padding_mode=padding_mode
            )
        else:
            self.out_conv = nn.Conv2d(
                features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
            )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, groups=self.groups, padding_mode=padding_mode)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, groups=self.groups, padding_mode=padding_mode)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners, antialias=True) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners, antialias=True)