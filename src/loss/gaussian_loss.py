from torch import nn
from loss.utils import LOSS_FN_DICT


DEPTH_REGULARIZATION_DICT = {
    "l1" : lambda x: x.log1p().mean(),
    "l2" : lambda x: x.log1p().square().mean()
}

def get_gaussian_train_loss(gaussian_loss_cfg):
    """
    Get the loss function based on the configuration.
    
    Args:
        gaussian_loss_cfg: Configuration containing loss parameters.
        
    Returns:
        GaussianLoss: An instance of the GaussianLoss class with specified parameters.
    """
    return GaussianLoss(
        loss_types=gaussian_loss_cfg.loss_types,
        loss_factors=gaussian_loss_cfg.loss_factors,
        depth_regularization=gaussian_loss_cfg.depth_regularization.type,
        depth_regularization_threshold=gaussian_loss_cfg.depth_regularization.threshold,
        depth_regularization_factor=gaussian_loss_cfg.depth_regularization.factor,
    )

class GaussianLoss(nn.Module):
    """
        Gaussian loss module for training 3dgs model.
    Args:
        loss_types (list): List of loss function names to be used.
        loss_factors (list): Corresponding factors for each loss function.
        diff_ops (list): List of differential operations to apply to the images.
        diff_factors (list): Corresponding factors for each differential operation.
    """
    def __init__(
        self,
        loss_types = [
            "l1_loss",
            "ssim"
        ],
        loss_factors = [
            0.8,
            0.2
        ],

        depth_regularization = "l2",
        depth_regularization_threshold = 15,
        depth_regularization_factor = 0.1,
    ):
        super().__init__()

        self.loss_fn = [
            LOSS_FN_DICT[ls] for ls in loss_types
        ]
        self.loss_factors = loss_factors

        self.depth_regularization = depth_regularization is not None
        if self.depth_regularization:
            def depth_regularization_fn(depth):
                return DEPTH_REGULARIZATION_DICT[depth_regularization](depth.sub(depth_regularization_threshold).relu())
            self.depth_regularization_fn = depth_regularization_fn
            self.depth_regularization_factor = depth_regularization_factor


    def forward(self, network_output, reference, estimated_depth):
        """
        Forward pass to compute the gaussian loss.
        Args:
            network_output (torch.Tensor): The output from the network, expected shape (B, C, H, W).
            reference (torch.Tensor): The ground truth image, expected shape (B, H, W).
            scale (torch.Tensor): The scale of gaussians.
        Returns:
            torch.Tensor: The computed gaussian loss.
        """
        assert len(network_output.shape) == len(reference.shape) == 4, \
            f"Input tensors must be 4D (B, C, H, W), got {network_output.shape} and {reference.shape}"
        reference = reference.clamp(0, 1)
        loss = 0
        for f, fn in zip(self.loss_factors, self.loss_fn):
            loss = loss + f * fn(network_output, reference)
        
        if self.depth_regularization:
            loss = loss + self.depth_regularization_factor * self.depth_regularization_fn(estimated_depth)

        return loss