from einops import repeat
from torch import nn
import torch

class Sobel(nn.Module):
    """
    Sobel operation for image processing tasks.
    """

    def __init__(self, img_channels=3):
        super().__init__()
        self.img_channels = img_channels

        self.sobel = nn.Conv2d(
            img_channels,
            img_channels * 2,
            3,
            groups=img_channels,
            bias=False,
            padding="same",
            padding_mode="zeros"
        )
        kernel = torch.tensor(
            [
                [
                    [-1,0,1],
                    [-2,0,2],
                    [-1,0,1]
                ],
                [
                    [-1,-2,-1],
                    [0,0,0],
                    [1,2,1]
                ]
            ],
            dtype=self.sobel.weight.dtype
        )
        kernel = repeat(kernel, "out_c_per_group h w -> (groups out_c_per_group) 1 h w", groups=img_channels)
        self.sobel.weight = torch.nn.Parameter(kernel, False)


    def forward(self, img):
        """
        Forward pass to compute Sobel.
        
        Args:
            img (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Computed Sobel, channel dimension is doubled to include both x and y gradients.
        """
        # B, C, H, W = img.shape
        # assert C == self.img_channels, f"Expected {self.img_channels} channels, got {C} channels."

        return self.sobel(img)

class Laplacian(nn.Module):
    """
    Laplacian operation for image processing tasks.
    """

    def __init__(self, img_channels=3):
        super().__init__()
        self.img_channels = img_channels
        self.lpl = nn.Conv2d(
            img_channels,
            img_channels,
            3,
            groups=img_channels,
            bias=False,
            padding="same",
            padding_mode="zeros"
        )
        kernel = torch.tensor(
            [[-1,-1,-1],
            [-1,8,-1],
            [-1,-1,-1]],
            dtype=self.lpl.weight.dtype
        )
        kernel = repeat(kernel, "h w -> groups 1 h w", groups=img_channels)
        self.lpl.weight = torch.nn.Parameter(kernel, False)

    def forward(self, img):
        """
        Forward pass to compute Laplacian.
        
        Args:
            img (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Computed Laplacian.
        """
        # B, C, H, W = img.shape
        # assert C == self.img_channels, f"Expected {self.img_channels} channels, got {C} channels."

        return self.lpl(img)

