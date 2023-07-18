# -*- encoding: utf-8 -*-
"""
File collate.py
Created on 2023/7/10 21:53
Copyright (c) 2023/7/10
@author: 
"""
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import numpy as np
import random

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 重点
def multi_pipeline_collate_fn(batch, samples_per_gpu: int = 1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size. This is designed to support the case that the
    :func:`__getitem__` of dataset return more than one images, such as
    query_support dataloader. The main difference with the :func:`collate_fn`
    in mmcv is it can process list[list[DataContainer]].

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases:

    1. cpu_only = True, e.g., meta data.
    2. cpu_only = False, stack = True, e.g., images tensors.
    3. cpu_only = False, stack = False, e.g., gt bboxes.

    Args:
        batch (list[list[:obj:`mmcv.parallel.DataContainer`]] |
            list[:obj:`mmcv.parallel.DataContainer`]): Data of
            single batch.
        samples_per_gpu (int): The number of samples of single GPU.
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    # This is usually a case in query_support dataloader, which
    # the :func:`__getitem__` of dataset return more than one images.
    # Here we process the support batch data in type of
    # List: [ List: [ DataContainer]]
    if isinstance(batch[0], Sequence):
        samples_per_gpu = len(batch[0]) * samples_per_gpu
        batch = sum(batch, [])
    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [
            multi_pipeline_collate_fn(samples, samples_per_gpu)
            for samples in transposed
        ]
    elif isinstance(batch[0], Mapping):
        return {
            key: multi_pipeline_collate_fn([d[key] for d in batch],
                                           samples_per_gpu)
            for key in batch[0]
        }
    else:
        # print(123)
        return default_collate(batch)
