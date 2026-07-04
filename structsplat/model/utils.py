from torch import nn
import torch
from torch.nn import functional as F
import math
from structsplat.scheduler.wsd import WSDScheduler


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, groups=1, padding_mode='replicate'):
        super().__init__()
        self.conv_block = nn.Sequential(
            activation(),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=groups, padding_mode=padding_mode),
            activation(),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=groups, padding_mode=padding_mode),
        )
    def forward(self, x):
        out = self.conv_block(x)
        return out + x


def dropout_gaussian(xs, dim=0, p=0.1):
    # Get the target dimension length
    length = xs[0].shape[dim]
    
    # Compute number of elements to keep
    # int(10 * (1 - 0.1)) = 9
    keep_n = int(length * (1 - p))
    
    if keep_n == length:
        return xs
        
    if keep_n == 0:
        # If all are dropped, return empty tensors (adjust as needed)
        return [torch.empty_like(x).narrow(dim, 0, 0) for x in xs]  
    # 1. Generate random permuted indices and take the first keep_n
    # randperm generates a shuffled sequence [0, 1, ..., length-1]
    keep_indices = torch.randperm(length, device=xs[0].device)[:keep_n]
    
    # 2. (Optional) sort indices
    # If this dimension is time steps or ordered features, keep order
    keep_indices, _ = torch.sort(keep_indices)
    
    # 3. Use index_select to keep the selected data
    # This changes the tensor shape by physically removing rows
    out = [torch.index_select(x, dim, keep_indices) for x in xs]
    
    return out

def norm_exp(x):
    d = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_normed = x / d
    return x_normed * torch.expm1(d)

ACTIVATION_DICT = {
    "norm_exp": norm_exp,
    "exp": torch.exp,
    "relu": torch.relu,
    "softplus_beta-10": lambda x: F.softplus(x, beta=10.0),
    "softplus_beta-log2": lambda x: F.softplus(x, beta=math.log(2)),
    "scaled_softplus": lambda x: 2 * F.softplus(x, beta=2*math.log(2)),
    "elup1" : lambda x: F.elu(x) + 1,
    "norm": F.normalize,   # only for rotation, default dim is 1.
    "linear": lambda x: x,
    "add_0p5": lambda x: torch.add(x, 0.5),
    "mul_0p25": lambda x: torch.mul(x, 0.25),
    "sigmoid": torch.sigmoid,
    "sigmoid2m": lambda x: torch.sigmoid(x-2),
    "adjusted_sigmoid": lambda x: torch.sigmoid(4*x),
    "adjusted_sigmoiddm1": lambda x:2*torch.sigmoid(4*x) - 1,
    "sigmoiddm1": lambda x:2*torch.sigmoid(x) - 1,
    "sigmoidm0p5": lambda x:torch.sigmoid(x) - 0.5,
    "sigmoidd2mm1": lambda x:2*torch.sigmoid(x-2) - 1,
    "inv_log": lambda x: torch.log(1 + x),

    "norm_dim3" : lambda x: F.normalize(x, dim=3),
}

OPTIMIZER_DICT = {}

SCHEDULER_DICT = {
    "WSD": WSDScheduler
}