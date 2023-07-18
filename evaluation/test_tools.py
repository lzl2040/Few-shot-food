# -*- encoding: utf-8 -*-
"""
File test_tools.py
Created on 2023/7/13 21:19
Copyright (c) 2023/7/13
@author: 
"""
import torch
from torch.utils.data import DataLoader
import torch.distributed as distributed

from util.tools import label_wrapper
from evaluation.meta_test_parallel import MetaTestParallel
from util.tools import collect_results_cpu

import mmcv
from mmengine.utils import ProgressBar
import numpy as np
import copy
from typing import Dict, Optional, Union

# z scores of different confidence intervals
Z_SCORE = {
    0.50: 0.674,
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.98: 2.326,
    0.99: 2.576,
}

def multi_gpu_meta_test(model,
                        num_test_tasks: int,
                        support_dataloader: DataLoader,
                        query_dataloader: DataLoader,
                        meta_test_cfg: Optional[Dict] = None,
                        eval_kwargs: Optional[Dict] = None,
                        logger: Optional[object] = None,
                        confidence_interval: float = 0.95,
                        show_task_results: bool = False) -> Dict:
    """Distributed meta testing on multiple gpus.

    During meta testing, model might be further fine-tuned or added extra
    parameters. While the tested model need to be restored after meta
    testing since meta testing can be used as the validation in the middle
    of training. To detach model from previous phase, the model will be
    copied and wrapped with :obj:`MetaTestParallel`. And it has full
    independence from the training model and will be discarded after the
    meta testing.

    In the distributed situation, the :obj:`MetaTestParallel` on each GPU
    is also independent. The test tasks in few shot leaning usually are very
    small and hardly benefit from distributed acceleration. Thus, in
    distributed meta testing, each task is done in single GPU and each GPU
    is assigned a certain number of tasks. The number of test tasks
    for each GPU is ceil(num_test_tasks / world_size). After all GPUs finish
    their tasks, the results will be aggregated to get the final result.

    Args:
        model (:obj:`MMDistributedDataParallel`): Model to be meta tested.
        num_test_tasks (int): Number of meta testing tasks.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            query data.
        test_set_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            all test data. Default: None.
        meta_test_cfg (dict): Config for meta testing. Default: None.
        eval_kwargs (dict): Any keyword argument to be used for evaluation.
            Default: None.
        logger (logging.Logger | None): Logger used for printing
            related information during evaluation. Default: None.
        confidence_interval (float): Confidence interval. Default: 0.95.
        show_task_results (bool): Whether to record the eval result of
            each task. Default: False.

    Returns:
        dict | None: Dict of meta evaluate results, containing `accuracy_mean`
            and `accuracy_std` of all test tasks.
    """
    # assert confidence_interval in Z_SCORE.keys()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    # Note that each task is tested on a single GPU. Thus the data and model
    # on different GPU should be independent. :obj:`MMDistributedDataParallel`
    # always automatically synchronizes the grad in different GPUs when doing
    # the loss backward, which can not meet the requirements. Thus we simply
    # copy the module and wrap it with an :obj:`MetaTestParallel`, which will
    # send data to the device model.
    model = MetaTestParallel(copy.deepcopy(model.module))

    # for the backbone-fixed methods, the features can be pre-computed
    # and saved in dataset to achieve acceleration
    # print_log('start meta testing', logger=logger)
    # prepare for meta test
    model.before_meta_test(meta_test_cfg)

    results_list = []

    # tasks will be evenly distributed on each gpus
    sub_num_test_tasks = num_test_tasks // world_size
    sub_num_test_tasks += 1 if num_test_tasks % world_size != 0 else 0
    if rank == 0:
        prog_bar = ProgressBar(num_test_tasks)
    for i in range(sub_num_test_tasks):
        task_id = (i * world_size + rank)
        if task_id >= num_test_tasks:
            continue
        # set support and query dataloader to the same task by task id
        query_dataloader.dataset.set_task_id(task_id)
        support_dataloader.dataset.set_task_id(task_id)
        # test a task
        results, gt_labels = test_single_task(model, support_dataloader,
                                              query_dataloader, meta_test_cfg)
        # evaluate predict result
        eval_result = query_dataloader.dataset.evaluate(
            results, gt_labels,"accuracy")
        eval_result['task_id'] = task_id
        results_list.append(eval_result)
        if rank == 0:
            prog_bar.update(world_size)

    collect_results_list = collect_results_cpu(
        results_list, num_test_tasks, tmpdir=None)
    if rank == 0:
        if show_task_results:
            # the result of each task will be logged into logger
            for results in collect_results_list:
                msg = ' '.join([f'{k}: {results[k]}' for k in results.keys()])
                # print_log(msg, logger=logger)

        meta_eval_results = dict()
        # print_log(
        #     f'number of tasks: {len(collect_results_list)}', logger=logger)
        # get the average accuracy and std
        for k in collect_results_list[0].keys():
            if k == 'task_id':
                continue
            mean = np.mean([res[k] for res in collect_results_list])
            std = np.std([res[k] for res in collect_results_list])
            std = Z_SCORE[confidence_interval] * std / np.sqrt(num_test_tasks)
            meta_eval_results[f'{k}_mean'] = mean
            meta_eval_results[f'{k}_std'] = std
        return meta_eval_results
    else:
        return None

def test_single_task(model: MetaTestParallel, support_dataloader: DataLoader,
                     query_dataloader: DataLoader, meta_test_cfg: Dict):
    """Test a single task.

    A task has two stages: handling the support set and predicting the
    query set. In stage one, it currently supports fine-tune based and
    metric based methods. In stage two, it simply forward the query set
    and gather all the results.

    Args:
        model (:obj:`MetaTestParallel`): Model to be meta tested.
        support_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            support data.
        query_dataloader (:obj:`DataLoader`): A PyTorch dataloader of
            query data.
        meta_test_cfg (dict): Config for meta testing.

    Returns:
        tuple:

            - results_list (list[np.ndarray]): Predict results.
            - gt_labels (np.ndarray): Ground truth labels.
    """
    # use copy of model for each task
    model = copy.deepcopy(model)
    # get ids of all classes in this task
    task_class_ids = query_dataloader.dataset.get_task_class_ids()

    # forward support set
    model.before_forward_support()
    support_cfg = meta_test_cfg.get('support', dict())
    # methods with fine-tune stage
    for i, data in enumerate(support_dataloader):
        # map input labels into range of 0 to numbers of classes-1
        data['gt_label'] = label_wrapper(data['gt_label'], task_class_ids)
        # forward in `support` mode
        model.forward(**data, mode='support')

    # forward query set
    model.before_forward_query()
    results_list, gt_label_list = [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(query_dataloader):
            gt_label_list.append(data.pop('gt_label'))
            # forward in `query` mode
            result = model.forward(**data, mode='query')
            results_list.extend(result)
        gt_labels = torch.cat(gt_label_list, dim=0).cpu().numpy()
    # map gt labels into range of 0 to numbers of classes-1.
    gt_labels = label_wrapper(gt_labels, task_class_ids)
    return results_list, gt_labels

