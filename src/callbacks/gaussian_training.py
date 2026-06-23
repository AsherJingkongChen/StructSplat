from matplotlib import pyplot as plt
from torchvision.utils import save_image
import pytorch_lightning as pl
import os
import torch


class VisualizationCallback(pl.Callback):
    def __init__(self, saving_dir:str, interval:int=100):
        """
            Callback to visualize splatting outputs during training.
            Args:
                saving_dir: Directory to save visualizations.
                interval: Interval at which to save visualizations.
        """
        super().__init__()
        self.saving_dir = saving_dir
        self.interval = interval
        os.makedirs(self.saving_dir, exist_ok=True)


    @torch.no_grad()
    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if (batch_idx+1) % self.interval == 0 or batch_idx == 0:
            img, _, src_number = batch
            B, S, C, H, W = img.shape
            S_src = src_number.unique().item()
            S_tar = S - S_src
            
            src, tar = torch.split(img, [S_src, S_tar], dim=1)
            rendered_images = outputs["rendered_images"].float()
            rendered_depth = outputs["rendered_depth"].squeeze(1).float().cpu().numpy()
            estimated_depth = outputs["estimated_depth"].float().cpu().numpy()
            gaussians = outputs["gaussians"]
            for b in range(B):
                vis_dir = os.path.join(self.saving_dir, "visualization", f"step-{str(batch_idx).zfill(5)}_rank-{trainer.global_rank}_group-{str(b).zfill(2)}")
                os.makedirs(vis_dir, exist_ok=True)
                try:
                    _plot_hist(gaussians, b, vis_dir)
                except Exception as e:
                    print(f"Error plotting histogram: {e}")
                str_length = len(str(S_src-1))    
                for s_src in range(S_src):
                    group_dir = os.path.join(vis_dir, f"src-{str(s_src).zfill(str_length)}")
                    os.makedirs(group_dir, exist_ok=True)
                    _save_images(
                        group_dir, 
                        src[b,s_src],
                        rendered_images[b*S+s_src], 
                        rendered_depth[b*S+s_src],
                        estimated_depth[b, s_src],            
                    )
                str_length = len(str(S_tar-1))
                for s_tar in range(S_tar):
                    group_dir = os.path.join(vis_dir, f"tar-{str(s_tar).zfill(str_length)}")
                    os.makedirs(group_dir, exist_ok=True)
                    _save_images(
                        group_dir, 
                        tar[b,s_tar],
                        rendered_images[b*S+S_src+s_tar], 
                        rendered_depth[b*S+S_src+s_tar],              
                    )


@torch.no_grad()
def _save_images(group_dir, img, rendered_image, rendered_depth, estimated_depth=None):  
    im_name = os.path.join(group_dir, "View.png")
    save_image(img.clip(0,1), im_name)
    
    im_name = os.path.join(group_dir, "RenderedImage.png")
    save_image(rendered_image.clip(0,1), im_name)

    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(rendered_depth)
    fig.colorbar(im)
    im_name = os.path.join(group_dir, "RenderedDepth.png")
    plt.savefig(im_name)
    plt.close()

    if estimated_depth is not None:
        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.imshow(estimated_depth)
        fig.colorbar(im)
        im_name = os.path.join(group_dir, "EstimatedDepth.png")
        plt.savefig(im_name)
        plt.close()


@torch.no_grad()
def _plot_hist(gaussians, b, vis_dir):
    for key, value in gaussians.items():
        if key == "raw_rotation":
            key = "raw_rotation_l2norm"
            value = value.norm(dim=1)
        plt.figure()
        plt.hist(value[b].flatten().float().cpu().numpy(), bins=100)
        plt.xlabel(key)
        hist_key_path = os.path.join(vis_dir, f"{key}.png")
        plt.savefig(hist_key_path)
        plt.close()