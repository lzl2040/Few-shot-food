# -*- encoding: utf-8 -*-
"""
File tools.py
Created on 2023/7/7 15:04
Copyright (c) 2023/7/7
@author: 
"""
import numpy as np
from contextlib import contextmanager
from typing import List, Union

import torch
from torch import Tensor

def list_from_file(path):
    f_list = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            f_list.append(line)
    return f_list

@contextmanager
def local_numpy_seed(seed):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def label_wrapper(labels: Union[Tensor, np.ndarray, List],
                  class_ids: List[int]) -> Union[Tensor, np.ndarray, list]:
    """Map input labels into range of 0 to numbers of classes-1.

    It is usually used in the meta testing phase, in which the class ids are
    random sampled and discontinuous.

    Args:
        labels (Tensor | np.ndarray | list): The labels to be wrapped.
        class_ids (list[int]): All class ids of labels.

    Returns:
        (Tensor | np.ndarray | list): Same type as the input labels.
    """
    class_id_map = {class_id: i for i, class_id in enumerate(class_ids)}
    if isinstance(labels, torch.Tensor):
        wrapped_labels = torch.tensor(
            [class_id_map[label.item()] for label in labels])
        wrapped_labels = wrapped_labels.type_as(labels).to(labels.device)
    elif isinstance(labels, np.ndarray):
        wrapped_labels = np.array([class_id_map[label] for label in labels])
        wrapped_labels = wrapped_labels.astype(labels.dtype)
    elif isinstance(labels, (tuple, list)):
        wrapped_labels = [class_id_map[label] for label in labels]
    else:
        raise TypeError('only support torch.Tensor, np.ndarray and list')
    return wrapped_labels
