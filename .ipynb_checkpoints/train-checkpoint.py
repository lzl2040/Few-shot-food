# -*- encoding: utf-8 -*-
"""
File train.py
Created on 2023/7/7 16:37
Copyright (c) 2023/7/7
@author: 
"""
from mmpretrain.evaluation import Accuracy
from model.losses import CrossEntropyLoss
from model.meta_baseline_head import MetaBaselineHead
from model.meta_baseline import MetaBaseline
import torch.optim as optim
from loguru import logger
import json
from datasets.food101_set import Food101Dataset
from datasets.dataset_wrapper import EpisodicDataset
from torch.utils.data import Dataset,DataLoader
from util.collate import multi_pipeline_collate_fn as collate
from functools import partial

if __name__ == '__main__':
    # 加载配置文件
    with open("./config/meta_baseline_config.json", 'r', encoding='utf-8') as f:
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
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs['train']['batch_size'],
        num_workers=0,
        pin_memory=False,
        collate_fn=partial(collate, samples_per_gpu=1)
    )
    # 创建network
    net = MetaBaseline(backbone="resnet50",head="meta_baseline_head")
    # 创建优化器
    optimzer = optim.Adam(net.parameters(), lr = 1e-4, weight_decay = 1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimzer, milestones=[40,80], gamma = 0.1)
    # 开始训练
    for epoch in range(1,configs['train']['epoches'] + 1):
        logger.info(f"epoch:{epoch}")
        for data in train_data_loader:
            optimzer.zero_grad()
            # 输入到模型里面
            loss = net.forward_train(data['support_data'], data['query_data'])
            print(loss)
            # 后向传播
            loss['loss'].backward()
            # 更新参数
            optimzer.step()
            scheduler.step()




