import os
from model.gaussian_wrapper import build_gaussian_wrapper
from data.gaussian_data import DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks.gaussian_training import VisualizationCallback


def train_gaussian(head_dict, cfg):
    """
    Train the gaussian prediction model using the provided configuration.
    
    Args:
        head_dict: Dictionary containing pretrained submodule states.
        cfg: Configuration object containing training parameters and settings.
        
    Returns:
        model: The trained gaussian prediction model.
    """

    # Initialize the model
    model:pl.LightningModule = build_gaussian_wrapper(head_dict, cfg)

    # Set up the logger
    logger_dir = os.path.join("logger", cfg.exp_name, cfg.key_word)
    os.makedirs(logger_dir, exist_ok=True)
    logger = CSVLogger(
        name=cfg.exp_name,
        save_dir=logger_dir,
    )
    

    # Set callbacks for saving checkpoints and visualizations
    saving_dir=os.path.join(
            cfg.gaussian_training_stage.recording.saving_dir, 
            cfg.exp_name,
            "gaussian_training_stage",
            cfg.key_word
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            saving_dir,
            "ckpt"
        ),
        filename="model-{step}",
        save_last=True,                  
        save_top_k=-1,
        every_n_train_steps=cfg.gaussian_training_stage.recording.ckpt_interval_steps
    )

    visualization_callback = VisualizationCallback(
        saving_dir=saving_dir,
        interval=cfg.gaussian_training_stage.recording.visualization_interval_steps,
    )

    deepspeed_config = cfg.gaussian_training_stage.deepspeed_config.to_dict()
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=cfg.gaussian_training_stage.total_steps,
        accelerator="gpu",
        devices="auto",
        strategy=DeepSpeedStrategy(config=deepspeed_config),
        logger=logger,
        callbacks=[checkpoint_callback, visualization_callback],
    )

    data_module = DataModule(cfg)

    gaussian_ckpt = cfg.gaussian_training_stage.init_ckpt.gaussian
    trainer.fit(model, datamodule=data_module,
            ckpt_path = gaussian_ckpt if gaussian_ckpt is not None and gaussian_ckpt.endswith(".ckpt") else None
            )

    return model