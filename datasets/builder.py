# -*- encoding: utf-8 -*-
"""
File builder.py
Created on 2023/7/14 0:49
Copyright (c) 2023/7/14
@author: 
"""
from torch.utils.data import DataLoader, Dataset
from util.collate import multi_pipeline_collate_fn as collate
from functools import partial
import copy
def build_meta_test_dataloader(dataset, meta_test_cfg):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        meta_test_cfg (dict): Config of meta testing.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        tuple[:obj:`Dataloader`]: `support_data_loader`, `query_data_loader`
            and `test_set_data_loader`.
    """
    support_batch_size = meta_test_cfg["support"]['batch_size']
    query_batch_size = meta_test_cfg["query"]['batch_size']
    num_support_workers = meta_test_cfg["support"]['num_workers']
    num_query_workers = meta_test_cfg["query"]['num_workers']

    support_data_loader = DataLoader(
        copy.deepcopy(dataset).support(),
        batch_size=support_batch_size,
        num_workers=num_support_workers,
        collate_fn=partial(collate, samples_per_gpu=support_batch_size),
        pin_memory=False,
        shuffle=True,
        drop_last=False)
    query_data_loader = DataLoader(
        copy.deepcopy(dataset).query(),
        batch_size=query_batch_size,
        num_workers=num_query_workers,
        collate_fn=partial(collate, samples_per_gpu=query_batch_size),
        pin_memory=False,
        shuffle=False)
    # # build test set dataloader for fast test
    # if meta_test_cfg['fast_test'] == 0:
    #     all_batch_size = meta_test_cfg.test_set.get('batch_size', 16)
    #     num_all_workers = meta_test_cfg.test_set.get('num_workers', 1)
    #     test_set_data_loader = DataLoader(
    #         copy.deepcopy(dataset).test_set(),
    #         batch_size=all_batch_size,
    #         num_workers=num_all_workers,
    #         collate_fn=partial(collate, samples_per_gpu=all_batch_size),
    #         pin_memory=False,
    #         shuffle=False)
    # else:
    #     test_set_data_loader = None
    return support_data_loader, query_data_loader