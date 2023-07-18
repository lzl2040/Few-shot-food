# -*- encoding: utf-8 -*-
"""
File test.py
Created on 2023/7/15 21:29
Copyright (c) 2023/7/15
@author:
多GPU情况下测试
"""
import argparse
import json
from loguru import logger

import torch
import torch.distributed as distributed
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.nn as nn

from datasets.builder import build_meta_test_dataloader
from datasets.dataset_wrapper import MetaTestDatastet
from datasets.food101_set import Food101Dataset
from evaluation.test_tools import multi_gpu_meta_test
from model.meta_baseline import MetaBaseline
from util.model_weights import load_weights

parser = argparse.ArgumentParser()
parser.add_argument('—-local_rank', type=int, default=0)
distributed.init_process_group(backend="nccl")
logger.info(f'CUDA Device count: {torch.cuda.device_count()}')
local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
logger.info(f'I am rank {local_rank} in this world of size {world_size}!')
torch.cuda.set_device(local_rank)

"""
CUDA_VISIBLE_DEVICES=1,2,3,4 OMP_NUM_THREADS=4 /home/gaoxingyu/anaconda3/envs/foodfewshot/bin/python -m torch.distributed.launch --master_port 9843 --nproc_per_node=4 test.py
"""

if __name__ == '__main__':
    # 加载配置文件
    with open("./config/meta_baseline_config.json", 'r', encoding='utf-8') as f:
        f = f.read()
        configs = json.loads(f)
        logger.info(f"Experiment Setting:{configs}")
    ## test dataset
    test_food_dataset = Food101Dataset(data_prefix=configs['test']['dataset']['data_prefix'],
                                       subset="test", classes=configs['test']['dataset']['classes'],
                                       img_size=configs['test']['dataset']['img_size'],
                                       ann_file=configs['test']['dataset']['ann'])
    ## test dataloader
    test_dataset = MetaTestDatastet(dataset=test_food_dataset,
                                    num_episodes=configs['test']['num_episodes'],
                                    num_ways=configs['test']['num_ways'],
                                    num_shots=configs['test']['num_shots'],
                                    num_queries=configs['test']['num_queries'],
                                    episodes_seed=configs['test']['episodes_seed'])
    meta_test_cfg = configs['test']['meta_test_cfg']
    test_support_dataloader, test_query_dataloader = build_meta_test_dataloader(test_dataset, meta_test_cfg)
    # 创建network
    net = nn.parallel.DistributedDataParallel(
        MetaBaseline(backbone="resnet50", head="meta_baseline_head").cuda(),
        device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    if configs['other']['weights_load_path']:
        net = load_weights(network=net, local_rank=local_rank,
                                                       weights_path=configs['other']['weights_load_path'],just_weight=True)
        logger.info("load trained weights")
    meta_test_results = multi_gpu_meta_test(
        net,
        meta_test_cfg['num_episodes'],
        test_support_dataloader,
        test_query_dataloader,
        meta_test_cfg)
    if local_rank == 0:
        logger.info(f"test result:{meta_test_results}")
