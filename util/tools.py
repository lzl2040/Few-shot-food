# -*- encoding: utf-8 -*-
"""
File tools.py
Created on 2023/7/7 15:04
Copyright (c) 2023/7/7
@author: 
"""
import random

import numpy as np
from contextlib import contextmanager
from typing import List, Union
import tempfile
import os
import os.path as osp
import shutil

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as distributed
import mmcv
import mmengine
from torch.nn.parallel._functions import Scatter as OrigScatter

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

def collect_results_cpu(result_part, size, tmpdir=None):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmengine.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        distributed.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmengine.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmengine.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    distributed.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmengine.load(part_file)
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
            # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

def scatter(inputs, target_gpus, dim=0):
    """Scatter inputs to target gpus.

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            if target_gpus != [-1]:
                return OrigScatter.apply(target_gpus, None, dim, obj)
            else:
                # for CPU inference we use self-implemented scatter
                # return Scatter.forward(target_gpus, obj)
                return None
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for targets in target_gpus]

    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary."""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

def make_file(names, path):
    img_root_path = "/home/gaoxingyu/dataset/food-101/images"
    with open(path,"w") as f:
        for name in names:
            img_dir = os.path.join(img_root_path,name)
            img_names = os.listdir(img_dir)
            sort_img_names = sorted(img_names,key=lambda s: int(s.split('.')[0]))
            for img_name in sort_img_names:
                img_path = os.path.join(img_dir,img_name).replace("/home/gaoxingyu/dataset/food-101/images/","")
                f.write(f"{img_path}\n")

def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "sigmoid",
    }
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))

def linear_init(module, activation="relu"):
    """Initialize a linear layer.

    Parameters
    ----------
    module : nn.Module
       module to initialize.

    activation : `torch.nn.modules.activation` or str, optional
        Activation that will be used on the `module`.
    """
    x = module.weight

    if module.bias is not None:
        module.bias.data.zero_()

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity="leaky_relu")
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity="relu")
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))

def weights_init(module, **kwargs):
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    module.is_resetted = True
    for m in module.modules():
        try:
            if hasattr(module, "reset_parameters") and module.is_resetted:
                # don't reset if resetted already (might want special)
                continue
        except AttributeError:
            pass

        if isinstance(m, torch.nn.modules.conv._ConvNd):
            # used in https://github.com/brain-research/realistic-ssl-evaluation/
            nn.init.kaiming_normal_(m.weight, mode="fan_out", **kwargs)
        elif isinstance(m, nn.Linear):
            linear_init(m, **kwargs)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def generate_split_food101dataset():
    class_path = "/home/gaoxingyu/dataset/food-101/meta/classes.txt"
    class_list = list_from_file(class_path)
    id2class = {i : _class for i, _class in enumerate(class_list)}
    # 选择71个类作为训练集的，30个作为测试的
    train_class_ids = random.sample(range(len(class_list)),71)
    test_class_ids = []
    for id in range(len(class_list)):
        if id not in train_class_ids:
            test_class_ids.append(id)
    # 切分
    train_class_name = [id2class[id] for id in train_class_ids]
    test_class_name = [id2class[id] for id in test_class_ids]
    # 顺序排序
    train_class_name = sorted(train_class_name)
    test_class_name = sorted(test_class_name)
    with open("/home/gaoxingyu/dataset/food-101/meta/fsl_train_class.txt","w") as f:
        for cls_name in train_class_name:
            f.write(f"{cls_name}\n")

    with open("/home/gaoxingyu/dataset/food-101/meta/fsl_test_class.txt","w") as f:
        for cls_name in test_class_name:
            f.write(f"{cls_name}\n")

    # 将这些数据保存在fsl_train.txt中,格式为:class_name/img_name
    make_file(train_class_name,"/home/gaoxingyu/dataset/food-101/meta/fsl_train.txt")
    make_file(test_class_name,"/home/gaoxingyu/dataset/food-101/meta/fsl_test.txt")

    return train_class_ids

if __name__ == '__main__':
    print(generate_split_food101dataset())
