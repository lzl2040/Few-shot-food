# -*- encoding: utf-8 -*-
"""
File dataset_wrapper.py
Created on 2023/7/7 11:48
Copyright (c) 2023/7/7
@author: 
"""
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset,DataLoader
from functools import partial
import os.path as osp
from typing import Mapping
from datasets.food101_set import Food101Dataset
from util import tools
import json
from util.collate import multi_pipeline_collate_fn as collate

class EpisodicDataset:
    """A wrapper of episodic dataset.

    It will generate a list of support and query images indices for each
    episode (support + query images). Every call of `__getitem__` will fetch
    and return (`num_ways` * `num_shots`) support images and (`num_ways` *
    `num_queries`) query images according to the generated images indices.
    Note that all the episode indices are generated at once using a specific
    random seed to ensure the reproducibility for same dataset.

    Args:
        dataset (:obj:`Dataset`): The dataset to be wrapped.
        num_episodes (int): Number of episodes. Noted that all episodes are
            generated at once and will not be changed afterwards. Make sure
            setting the `num_episodes` larger than your needs.
        num_ways (int): Number of ways for each episode.
        num_shots (int): Number of support data of each way for each episode.
        num_queries (int): Number of query data of each way for each episode.
        episodes_seed (int | None): A random seed to reproduce episodic
            indices. If seed is None, it will use runtime random seed.
            Default: None.
    """

    def __init__(self,
                 dataset: Dataset,
                 num_episodes: int,
                 num_ways: int,
                 num_shots: int,
                 num_queries: int,
                 episodes_seed: int):
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.num_episodes = num_episodes
        self._len = len(self.dataset)
        self.CLASSES = dataset.CLASSES
        # using same episodes seed can generate same episodes for same dataset
        # it is designed for the reproducibility of meta train or meta test
        self.episodes_seed = episodes_seed
        self.episode_idxes, self.episode_class_ids = \
            self.generate_episodic_idxes()

    def generate_episodic_idxes(self):
        """Generate batch indices for each episodic."""
        episode_idxes, episode_class_ids = [], []
        class_ids = [i for i in range(len(self.CLASSES))]
        # using same episodes seed can generate same episodes for same dataset
        # it is designed for the reproducibility of meta train or meta test
        with tools.local_numpy_seed(self.episodes_seed):
            for _ in range(self.num_episodes):
                np.random.shuffle(class_ids)
                # sample classes
                sampled_cls = class_ids[:self.num_ways]
                episode_class_ids.append(sampled_cls)
                episodic_support_idx = []
                episodic_query_idx = []
                # sample instances of each class
                for i in range(self.num_ways):
                    shots = self.dataset.sample_shots_by_class_id(
                        sampled_cls[i], self.num_shots + self.num_queries)
                    episodic_support_idx += shots[:self.num_shots]
                    episodic_query_idx += shots[self.num_shots:]
                episode_idxes.append({
                    'support': episodic_support_idx,
                    'query': episodic_query_idx
                })
        return episode_idxes, episode_class_ids

    def __getitem__(self, idx: int):
        """Return a episode data at the same time.

        For `EpisodicDataset`, this function would return num_ways *
        num_shots support images and num_ways * num_queries query image.
        """
        support_data = [self.dataset[i] for i in self.episode_idxes[idx]['support']]
        query_data = [self.dataset[i] for i in self.episode_idxes[idx]['query']]
        return {
            'support_data':support_data,
            'query_data':query_data
        }

    def __len__(self):
        """The length of the dataset is the number of generated episodes."""
        return self.num_episodes

    def evaluate(self, *args, **kwargs):
        """Evaluate prediction."""
        return self.dataset.evaluate(*args, **kwargs)

    def get_episode_class_ids(self, idx: int):
        """Return class ids in one episode."""
        return self.episode_class_ids[idx]

if __name__ == '__main__':
    # 加载配置文件
    with open("../config/meta_baseline_config.json",'r',encoding = 'utf-8') as f:
        f = f.read()
        configs = json.loads(f)
    # 创建数据集
    dataset = Food101Dataset(data_prefix=configs['dataset']['data_prefix'],
                             subset=configs['subset'], classes=configs['dataset']['classes'])
    train_dataset = EpisodicDataset(dataset=dataset,
                                    num_episodes=configs['train']['num_episodes'],
                                    num_ways=configs['train']['num_ways'],
                                    num_shots=configs['train']['num_shots'],
                                    num_queries=configs['train']['num_queries'],
                                    episodes_seed=configs['train']['episodes_seed'])
    # dataloader
    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs['train']['batch_size'],
        num_workers=0,
        pin_memory=False,
        collate_fn=partial(collate, samples_per_gpu=1)
    )
    for data in data_loader:
        print(data['support_data'])
        break

