# -*- encoding: utf-8 -*-
"""
File train.py
Created on 2023/7/7 16:37
Copyright (c) 2023/7/7
@author: 
"""
from model.attention_neural_process import AttentionNeuralProcess
from model.meta_baseline import MetaBaseline
from datasets.food101_set import Food101Dataset
from datasets.dataset_wrapper import EpisodicDataset,MetaTestDatastet
from util.collate import multi_pipeline_collate_fn as collate
from functools import partial
from util.model_weights import *
from datasets.builder import build_meta_test_dataloader
from evaluation.test_tools import multi_gpu_meta_test

import torch
import torch.distributed as distributed
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.nn as nn

import os
import time
import random
import numpy as np
from loguru import logger
import json
import argparse
import datetime

"""
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1,2,3,4 OMP_NUM_THREADS=4 /home/gaoxingyu/anaconda3/envs/foodfewshot/bin/python -m torch.distributed.launch --master_port 9843 --nproc_per_node=4 train.py
CUDA_VISIBLE_DEVICES=1,2,3,4 OMP_NUM_THREADS=4 nohup /home/gaoxingyu/anaconda3/envs/foodfewshot/bin/python -m torch.distributed.launch --master_port 9843 --nproc_per_node=4 train.py &
"""


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 31) + worker_id + local_rank * 100
    np.random.seed(worker_seed)
    random.seed(worker_seed)

"""
Initial setup
"""
parser = argparse.ArgumentParser()
parser.add_argument('—-local_rank', type=int, default=0)
distributed.init_process_group(backend="nccl")
logger.info(f'CUDA Device count: {torch.cuda.device_count()}')
local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
logger.info(f'I am rank {local_rank} in this world of size {world_size}!')
torch.cuda.set_device(local_rank)
# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

if __name__ == '__main__':
    # 加载配置文件
    with open("./config/meta_baseline_config.json", 'r', encoding='utf-8') as f:
        f = f.read()
        configs = json.loads(f)
        logger.info(f"Experiment Setting:{configs}")
    # 创建数据集
    ## train_dataset
    train_food_dataset = Food101Dataset(data_prefix=configs['train']['dataset']['data_prefix'],
                             subset="train", classes=configs['train']['dataset']['classes'],
                             img_size=configs['train']['dataset']['img_size'],ann_file=configs['train']['dataset']['ann'])
    train_dataset = EpisodicDataset(dataset=train_food_dataset,
                                    num_episodes=configs['train']['num_episodes'],
                                    num_ways=configs['train']['num_ways'],
                                    num_shots=configs['train']['num_shots'],
                                    num_queries=configs['train']['num_queries'],
                                    episodes_seed=configs['train']['episodes_seed'])
    ## train dataloader
    train_samper = torch.utils.data.distributed.DistributedSampler(train_dataset, rank = local_rank, shuffle=True)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs['train']['per_gpu_batch_size'],
        sampler=train_samper,
        num_workers=configs['train']['per_gpu_workers'],
        collate_fn=partial(collate, samples_per_gpu=1),
        worker_init_fn=worker_init_fn,
        drop_last=True
    )
    ## test dataset
    test_food_dataset = Food101Dataset(data_prefix=configs['test']['dataset']['data_prefix'],
                             subset="test", classes=configs['test']['dataset']['classes'],
                             img_size=configs['test']['dataset']['img_size'],ann_file=configs['test']['dataset']['ann'])
    ## test dataloader
    test_dataset = MetaTestDatastet(dataset=test_food_dataset,
                                    num_episodes=configs['test']['num_episodes'],
                                    num_ways=configs['test']['num_ways'],
                                    num_shots=configs['test']['num_shots'],
                                    num_queries=configs['test']['num_queries'],
                                    episodes_seed=configs['test']['episodes_seed'])
    meta_test_cfg = configs['test']['meta_test_cfg']
    test_support_dataloader, test_query_dataloader = build_meta_test_dataloader(test_dataset,meta_test_cfg)
    # 创建network
    if configs["train_model"] == 'meta_baseline':
        net_noraml = MetaBaseline(backbone="resnet50",head="meta_baseline_head")
    elif configs["train_model"] == "anp":
        net_noraml = AttentionNeuralProcess(backbone="resnet50",head="anp_head",class_num=configs['train']['num_ways'])
    net = nn.parallel.DistributedDataParallel(
        net_noraml.cuda(),device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    # 创建优化器
    optimzer = optim.AdamW(net.parameters(), lr = configs['optimizer']['lr'], weight_decay = configs['optimizer']['wd'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimzer, milestones=[80 * 2000], gamma = 0.5)
    # 加载训练好的权重
    epoch = 1
    if configs['other']['weights_load_path']:
        net, optimzer, scheduler, epoch = load_weights(network=net, optimzer=optimzer,
                                                       scheduler=scheduler,local_rank=local_rank,
                                                       weights_path=configs['other']['weights_load_path'])
    # 开始训练
    total_interations = 0
    # 获得训练开始的时间
    now_time = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
    save_path = os.path.join(configs['other']['weights_save_path'], now_time)
    total_train_iter = len(train_data_loader)
    # 最大的准确率
    best_top1_acc = 0
    best_top1_acc_epoch = 1
    # Need this to select random bases in different workers
    np.random.seed(np.random.randint(2 ** 30 - 1) + local_rank * 100)
    while epoch <= configs['train']['epoches']:
        logger.info(f"Current epoch:{epoch}")
        train_samper.set_epoch(epoch)
        episode_num = 0
        net.train()
        start_time = datetime.datetime.now()
        for data in train_data_loader:
            optimzer.zero_grad(set_to_none=True)
            # 输入到模型里面
            loss = net.module.forward_train(data['support_data'], data['query_data'])
            # 后向传播
            loss['loss'].backward()
            # 更新参数
            optimzer.step()
            scheduler.step()
            total_interations += 1
            episode_num += 1
            if total_interations % configs['other']['log_interval'] == 0:
                end_time = datetime.datetime.now()
                consume_time = (end_time - start_time).seconds
                if local_rank == 0:
                    logger.info(f"[Epoch:{epoch} Interation/Total:{episode_num}/{total_train_iter}] Lr:{scheduler.get_last_lr()[0]:.5f} Loss:{loss['loss'].item():.5f} Time:{consume_time:.5f}")
                start_time = datetime.datetime.now()
        if epoch % configs["other"]["save_interval"] == 0:
            logger.info(f"save model_{epoch}")
            save_weights(network=net, optimizer=optimzer, scheduler=scheduler, epoch = epoch,save_path = save_path)
        if epoch % configs['other']['val_interval'] == 0:
            # 进行测试
            meta_test_results = multi_gpu_meta_test(
                net,
                meta_test_cfg['num_episodes'],
                test_support_dataloader,
                test_query_dataloader,
                meta_test_cfg)
            if local_rank == 0:
                logger.info(f"test result:{meta_test_results}")
                top1_acc = meta_test_results['accuracy_top-1_thr_0.00_mean']
                if top1_acc > best_top1_acc:
                    best_top1_acc = top1_acc
                    best_top1_acc_epoch = epoch
                    # 保存最好的模型
                    save_weights(network=net,save_path=save_path,save_best=True)
        epoch += 1
    logger.info(f"best top1 accuracy: {best_top1_acc} achieves in epoch{best_top1_acc_epoch}")
    distributed.destroy_process_group()




