from torch.nn.functional import binary_cross_entropy , l1_loss, mse_loss
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as lpips

LOSS_FN_DICT = {
    "ce_loss" : binary_cross_entropy,
    "l1_loss" : l1_loss,
    "l2_loss" : mse_loss,
    "ssim" : lambda img1, img2: 1.0 - ssim(img1, img2, data_range=(0.0, 1.0)),
    "lpips" : lambda img1, img2: lpips(img1.clamp(0, 1), img2.clamp(0, 1), normalize=True),
}