from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
import pytorch_lightning as pl
import json
from data.utils import DataLoaderX


def read_train_scenes_(cfg):
    cfg.gaussian_training_stage.data.scenes = []
    for js_path in cfg.gaussian_training_stage.data.annotation:
        with open(js_path) as f:
            cfg.gaussian_training_stage.data.scenes.extend(json.load(f))

def read_test_scenes_(cfg):
    cfg.gaussian_evaluation_stage.data.scenes = []
    for js_path in cfg.gaussian_evaluation_stage.data.annotation:
        with open(js_path) as f:
            cfg.gaussian_evaluation_stage.data.scenes.extend(json.load(f))


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        read_train_scenes_(cfg)
        read_test_scenes_(cfg)

    def setup(self, stage=None):
        datasets = __import__(f"data.datasets.{self.cfg.gaussian_training_stage.data.name}", fromlist=["TrainDataSet"])
        TrainDataSet = getattr(datasets, "TrainDataSet")
        if stage == 'fit' or stage is None:
            self.train_dataset = TrainDataSet(self.cfg.gaussian_training_stage, self.trainer.world_size, self.trainer.global_rank)

        elif stage == 'test':
            datasets = __import__(f"data.datasets.{self.cfg.gaussian_evaluation_stage.data.name}", fromlist=["EvaluateDataSet"])
            EvaluateDataSet = getattr(datasets, "EvaluateDataSet")
            self.train_dataset = TrainDataSet(self.cfg.gaussian_training_stage, self.trainer.world_size, self.trainer.global_rank)
            self.test_dataset = EvaluateDataSet(self.cfg.gaussian_evaluation_stage)
            self.test_sampler = SequentialSampler(
                self.test_dataset, 
            )
  

    def train_dataloader(self):
        sampler = DistributedSampler(
            self.train_dataset,
            shuffle=False,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank
        )

        return DataLoaderX(
            self.train_dataset,
            batch_size = self.cfg.gaussian_training_stage.data.batch_size,
            num_workers = self.cfg.gaussian_training_stage.data.num_workers,
            sampler = sampler,
            shuffle = False,
            pin_memory = True
        )

    def test_dataloader(self):
        return DataLoaderX(
            self.test_dataset,
            batch_size = self.cfg.gaussian_evaluation_stage.data.batch_size,
            num_workers = self.cfg.gaussian_evaluation_stage.data.num_workers,
            sampler = self.test_sampler
        )
