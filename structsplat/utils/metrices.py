from einops import reduce
import torch
from skimage.metrics import structural_similarity
from lpips import LPIPS

class TestMetrics:
    def __init__(self, device):
        self.test_psnr_list = []
        self.test_ssim_list = []
        self.test_lpips_list = []
        self.lpips_model = LPIPS(net="vgg").to(device)

    def clear(self):
        self.test_psnr_list.clear()
        self.test_ssim_list.clear()
        self.test_lpips_list.clear()

    def update_metrics(
        self,
        ground_truth,
        predicted,
    ):
        psnr = self.compute_psnr(ground_truth, predicted)
        ssim = self.compute_ssim(ground_truth, predicted)
        lpips = self.compute_lpips(ground_truth, predicted)

        self.test_psnr_list.append(psnr.mean().cpu().item())
        self.test_ssim_list.append(ssim.mean().cpu().item())
        self.test_lpips_list.append(lpips.mean().cpu().item())

    def get_avg_metrics(self):
        if len(self.test_psnr_list) == 0 or len(self.test_ssim_list) == 0 or len(self.test_lpips_list) == 0:
            raise ValueError("No metrics to average. Make sure to call update_metrics() before get_avg_metrics().")
        
        avg_psnr = sum(self.test_psnr_list) / len(self.test_psnr_list)
        avg_ssim = sum(self.test_ssim_list) / len(self.test_ssim_list)
        avg_lpips = sum(self.test_lpips_list) / len(self.test_lpips_list)
        return avg_psnr, avg_ssim, avg_lpips

    @torch.no_grad()
    def compute_psnr(
        self,
        ground_truth,
        predicted,
    ):
        ground_truth = ground_truth.clip(min=0, max=1)
        predicted = predicted.clip(min=0, max=1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        return -10 * mse.log10()

    @torch.no_grad()
    def compute_ssim(
        self,
        ground_truth,
        predicted,
    ):
        ssim = [
            structural_similarity(
                gt.detach().cpu().numpy(),
                hat.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(ground_truth, predicted)
        ]
        return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)

    @torch.no_grad()
    def compute_lpips(
        self,
        ground_truth,
        predicted,
    ):
        value = self.lpips_model.forward(ground_truth, predicted, normalize=True)
        return value.squeeze()
