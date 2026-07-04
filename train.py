import argparse
import os
import torch
from structsplat.config import load_configs
from structsplat.train.train_gaussian import train_gaussian
from pytorch_lightning.utilities import rank_zero_only
import yaml


TRAIN_FN_DICT = {
    'gaussian_training_stage': train_gaussian
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', "-s", type=str, default='gaussian_training_stage',
                        help='Stage of experiment, e.g., gaussian_training_stage')
    parser.add_argument('--config', "-c", type=str, default='config/dl3dv.yaml')

    return parser.parse_known_args()

@rank_zero_only
def print_and_save_cfg(stage, cfg):
    print("*" * 31 + "Experiment Stage" + "*" * 31)
    print(stage)
    print("*" * 30 + "Experiment Setting" + "*" * 30)
    print(cfg)
    print("*" * 78)

    cfg_dir = os.path.join(
        cfg.gaussian_training_stage.recording.saving_dir, 
        cfg.exp_name,
        stage,
        cfg.key_word,
    )
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path =  os.path.join(cfg_dir, "config.yaml")

    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg.to_dict(), f)

def main():
    device = torch.device('cuda:0')
    torch.set_float32_matmul_precision('high')
    args, addition_args = parse_args()
    cfg = load_configs(args.config, addition_args)
    print_and_save_cfg(args.stage, cfg)
    if args.stage == "gaussian_training_stage":
        module_dict = {}
        
        if cfg.gaussian_training_stage.init_ckpt.gaussian is not None and cfg.gaussian_training_stage.init_ckpt.gaussian.endswith(".bin"):
            gaussian_wrapper_dict = torch.load(cfg.gaussian_training_stage.init_ckpt.gaussian)
            module_dict["gaussian_predictor"] = {}
            for k, v in gaussian_wrapper_dict.items():
                if k.startswith("gaussian_predictor."):
                    module_dict["gaussian_predictor"][k.removeprefix("gaussian_decoder.")] = v

        inp_args = (module_dict, cfg)
    else:
        inp_args = (cfg,)

    TRAIN_FN_DICT[args.stage](*inp_args)
    max_reserved = torch.cuda.max_memory_reserved(device)
    print(f"Peak GPU RAM Usage: {max_reserved / 1024**2:.2f} MB")


if __name__ == '__main__':
    main() 