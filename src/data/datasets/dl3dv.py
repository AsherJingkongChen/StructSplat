from torch.utils.data import Dataset
from copy import copy
import random
import os
import torch
from data.utils import load_and_preprocess_images


class TrainDataSet(Dataset):
    def __init__(self, training_stage_cfg, world_size, global_rank):
        super().__init__()
        self.start_idx = training_stage_cfg.data.start_idx if hasattr(training_stage_cfg.data, "start_idx") else 0
        self.scenes_backup: list = training_stage_cfg.data.scenes
        self.scenes = copy(self.scenes_backup)
        random.seed(self.start_idx//world_size)
        random.shuffle(self.scenes)
        self.tar_ratio = training_stage_cfg.data.tar_ratio

        self.min_src_number = training_stage_cfg.data.min_src_number
        self.max_src_number_range = training_stage_cfg.data.max_src_number_range
        assert self.min_src_number >= 2 or self.tar_ratio == 0
        assert self.min_src_number <= self.max_src_number_range[0] and self.max_src_number_range[0] <= self.max_src_number_range[1]
        
        self.total_steps = training_stage_cfg.total_steps
        self.distance_range_between_src_views = training_stage_cfg.data.distance_range_between_src_views
        self.increasing_frequency = training_stage_cfg.data.increasing_src_number_frequency

        self.img_number = training_stage_cfg.data.img_number \
            if hasattr(training_stage_cfg.data, "img_number") else None
        
        self.world_size = world_size
        self.global_rank = global_rank
        self.global_batch_size = training_stage_cfg.data.batch_size * self.world_size
        self.length = self.global_batch_size * self.total_steps
        self.crop = training_stage_cfg.data.resize.crop
        self.new_size = training_stage_cfg.data.resize.new_size

    def __getitem__(self, batch_idx):
        batch_idx = batch_idx + self.start_idx
        random.seed(batch_idx//self.world_size)
        if len(self.scenes) < self.world_size:
            self.scenes.clear()
            self.scenes.extend(self.scenes_backup)
            random.shuffle(self.scenes)

        img_number = float('inf')
        for i in range(self.world_size):
            checking_scene = self.scenes.pop()
            img_number = min(img_number, checking_scene["length"])
            if i == self.global_rank:
                scene = checking_scene

        scene_path = scene["path"]
        if self.img_number is not None:
            img_number = min(img_number, self.img_number)

        src_distance_range = self.distance_range_between_src_views
        step_idx = batch_idx // self.global_batch_size
        random.seed(step_idx)

        max_src_number = self.max_src_number_range[0] + (step_idx // self.increasing_frequency)
        max_src_number = min(max_src_number, (img_number - 1) // src_distance_range[1] + 1, self.max_src_number_range[1])
        src_number = random.randint(self.min_src_number, max_src_number)
        max_src_range = (src_number - 1) * src_distance_range[1] + 1


        img_pathes = [os.path.join(scene_path, img) for img in os.listdir(scene_path) if img.endswith((".JPG", ".jpg", ".PNG", ".png"))]
        img_pathes = sorted(img_pathes)[:img_number]

        random.seed(batch_idx)
        src_idx = [random.randint(0, img_number - max_src_range)]
        tar_idx = []
        for _ in range(src_number-1):
            src_distance = random.randint(*src_distance_range)
            tar_distance = src_distance / (self.tar_ratio + 1)
            for i in range(1, self.tar_ratio + 1):
                tar_idx.append(src_idx[-1] + round(i * tar_distance))

            src_idx.append(src_idx[-1] + src_distance)

        
        img_idx = src_idx + tar_idx
        try:
            img = load_and_preprocess_images([img_pathes[i] for i in img_idx], crop=self.crop, new_size=self.new_size)
        except Exception as e:
            raise Exception(f"Error loading images for scene {scene_path} with image idx {img_idx}: {e}") from e
        sorting_idx = torch.tensor(img_idx).argsort()
        return img, sorting_idx, src_number

    def __len__(self):
        return self.length


class EvaluateDataSet(Dataset):
    def __init__(self, evaluation_stage_cfg):
        super().__init__()
        self.scenes = evaluation_stage_cfg.data.scenes
        self.crop = evaluation_stage_cfg.data.resize.crop
        self.new_size = evaluation_stage_cfg.data.resize.new_size

    def __getitem__(self, batch_idx):
        scene = self.scenes[batch_idx]
        scene_dir = scene["path"]
        src = scene["src"]
        tar = scene["tar"]
        src_path = [os.path.join(scene_dir, s) for s in src]
        tar_path = [os.path.join(scene_dir, t) for t in tar]
        img_path = src_path + tar_path
        img = load_and_preprocess_images(img_path, crop=self.crop, new_size=self.new_size)

        indexed_path = list(enumerate(img_path))
        sorted_path = sorted(indexed_path, key=lambda x: x[1])
        sorting_idx = [index for index, _ in sorted_path]
        return img, torch.tensor(sorting_idx), scene_dir, len(src)

    def __len__(self):
        return len(self.scenes)