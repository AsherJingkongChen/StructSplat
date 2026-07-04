import argparse
from hashlib import md5
import os
import torch
from structsplat.model.gaussian_wrapper import build_gaussian_wrapper
from structsplat.config import load_configs
from structsplat.callbacks.gaussian_evaluation import RecordingAndVisualizationCallback
from structsplat.data.gaussian_data import DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_only
import yaml



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', "-c", type=str, default='config/dl3dv.yaml')

    return parser.parse_known_args()


def get_gaussian_wrapper(cfg, param_dict):
    module_dict = {
        "gaussian_predictor": {}
    }
    for k, v in param_dict.items():
        if module_dict.get("gaussian_predictor") is not None and k.startswith("gaussian_predictor."):
                module_dict["gaussian_predictor"][k.removeprefix("gaussian_predictor.")] = v
    wrapper = build_gaussian_wrapper(module_dict, cfg)

    for param in wrapper.parameters():
        param.requires_grad = False
    wrapper.eval()

    return wrapper


@rank_zero_only
def print_and_save_cfg(cfg,cfg_saving_dir):
    print("*" * 31 + "Experiment Stage" + "*" * 31)
    print("Gaussian Evaluation")
    print("*" * 30 + "Experiment Setting" + "*" * 30)
    print(cfg)
    print("*" * 78)

    os.makedirs(cfg_saving_dir, exist_ok=True)
    cfg_path =  os.path.join(cfg_saving_dir, "config.yaml")

    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg.to_dict(), f)


def main():
    device = torch.device('cuda:0')
    torch.set_float32_matmul_precision('high')
    args, addition_args = parse_args()
    cfg = load_configs(args.config, addition_args)
    saving_dir = os.path.join(
        cfg.gaussian_evaluation_stage.recording.saving_dir, 
        cfg.exp_name,
        "gaussian_evaluation_stage",
        cfg.key_word,
    )
    print_and_save_cfg(cfg, saving_dir)
    param_dict = torch.load(cfg.gaussian_evaluation_stage.ckpt)
    model = get_gaussian_wrapper(cfg, param_dict)

    logger_dir = os.path.join("logger", cfg.exp_name, cfg.key_word)
    os.makedirs(logger_dir, exist_ok=True)
    logger = CSVLogger(
        name=cfg.exp_name, 
        save_dir=logger_dir,
    )

    recording_and_visualization_callback = RecordingAndVisualizationCallback(
        saving_dir=saving_dir,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=1,
        accelerator="gpu",
        devices=[0],
        strategy="auto",
        callbacks=[
            recording_and_visualization_callback,
        ],
        logger=logger,
    )
    data_module = DataModule(cfg)
    trainer.test(model, data_module)
    max_reserved = torch.cuda.max_memory_reserved(device)
    print(f"Peak GPU RAM Usage: {max_reserved / 1024**2:.2f} MB")

if __name__ == '__main__':
    main() 