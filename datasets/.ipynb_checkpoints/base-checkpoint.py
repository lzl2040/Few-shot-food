# -*- encoding: utf-8 -*-
"""
File base.py
Created on 2023/7/7 11:51
Copyright (c) 2023/7/7
@author: 
"""
import copy
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Mapping, Optional, Sequence, Union
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from util import tools
import os.path as osp
from PIL import Image
import torch

class BaseFewShotDataset(Dataset, metaclass=ABCMeta):
    """Base few shot dataset.

    Args:
        data_prefix (str): The prefix of data path.
        pipeline (list): A list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`.
        classes (str | Sequence[str] | None): Classes for model training and
            provide fixed label for each class. Default: None.
        ann_file (str | None): The annotation file. When `ann_file` is str,
            the subclass is expected to read from the `ann_file`. When
            `ann_file` is None, the subclass is expected to read according
            to data_prefix. Default: None.
    """

    CLASSES = None

    def __init__(self,
                 data_prefix: str,
                 classes: Optional[Union[str, List[str]]] = None,
                 ann_file: Optional[str] = None) -> None:
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.pipeline = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations()

        # build a look up dict for sampling instances by class id
        self.data_infos_class_dict = {i: [] for i in range(len(self.CLASSES))}
        for idx, data_info in enumerate(self.data_infos):
            self.data_infos_class_dict[data_info['gt_label'].item()].append(
                idx)

    def load_image_from_file(self,info_dict):
        img_prefix = info_dict['img_prefix']
        img_name = info_dict['img_info']['filename']
        img_file = osp.join(img_prefix,f"{img_name}.jpg")
        img_data = Image.open(img_file).convert('RGB')
        return img_data

    @abstractmethod
    def load_annotations(self):
        pass

    @property
    def class_to_idx(self) -> Mapping:
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def prepare_data(self, idx: int) -> Dict:
        results = copy.deepcopy(self.data_infos[idx])
        imgs_data = self.load_image_from_file(results)
        # print(self.data_infos[idx]['gt_label'])
        data = {
            "img" : self.pipeline(imgs_data),
            "gt_label" : torch.tensor(self.data_infos[idx]['gt_label'])
        }
        # print(data)
        return data

    def sample_shots_by_class_id(self, class_id: int,
                                 num_shots: int) -> List[int]:
        """Random sample shots of given class id."""
        all_shot_ids = self.data_infos_class_dict[class_id]
        return np.random.choice(
            all_shot_ids, num_shots, replace=False).tolist()

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict:
        return self.prepare_data(idx)

    @classmethod
    def get_classes(cls,
                    classes: Union[Sequence[str],
                                   str] = None) -> Sequence[str]:
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): Three types of input
                will correspond to different processing logics:

                - If `classes` is a tuple or list, it will override the
                  CLASSES predefined in the dataset.
                - If `classes` is None, we directly use pre-defined CLASSES
                  will be used by the dataset.
                - If `classes` is a string, it is the path of a classes file
                  that contains the name of all classes. Each line of the file
                  contains a single class name.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """

        if isinstance(classes, str):
            # take it as a file path
            # mmcv.list_from_file(classes) classes为文件名,里面含有类的名字
            class_names = tools.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    # @staticmethod
    # def evaluate(results: List,
    #              gt_labels: np.array,
    #              metric: Union[str, List[str]] = 'accuracy',
    #              metric_options: Optional[dict] = None,
    #              logger: Optional[object] = None) -> Dict:
    #     """Evaluate the dataset.
    #
    #     Args:
    #         results (list): Testing results of the dataset.
    #         gt_labels (np.ndarray): Ground truth labels.
    #         metric (str | list[str]): Metrics to be evaluated.
    #             Default value is `accuracy`.
    #         metric_options (dict | None): Options for calculating metrics.
    #             Allowed keys are 'topk', 'thrs' and 'average_mode'.
    #             Default: None.
    #         logger (logging.Logger | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #
    #     Returns:
    #         dict: evaluation results
    #     """
    #     if isinstance(metric, str):
    #         metrics = [metric]
    #     else:
    #         metrics = metric
    #     allowed_metrics = [
    #         'accuracy', 'precision', 'recall', 'f1_score', 'support'
    #     ]
    #     eval_results = {}
    #     results = np.vstack(results)
    #     num_imgs = len(results)
    #     assert len(gt_labels) == num_imgs, \
    #         'dataset testing results should be of the same ' \
    #         'length as gt_labels.'
    #
    #     invalid_metrics = set(metrics) - set(allowed_metrics)
    #     if len(invalid_metrics) != 0:
    #         raise ValueError(f'metirc {invalid_metrics} is not supported.')
    #
    #     if metric_options is None:
    #         metric_options = {'topk': 1}
    #     topk = metric_options.get('topk', 1)
    #     thrs = metric_options.get('thrs', 0.0)
    #     average_mode = metric_options.get('average_mode', 'macro')
    #
    #     if 'accuracy' in metrics:
    #         # acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
    #         acc = 0
    #         if isinstance(topk, tuple):
    #             eval_results_ = {
    #                 f'accuracy_top-{k}': a
    #                 for k, a in zip(topk, acc)
    #             }
    #         else:
    #             eval_results_ = {'accuracy': acc}
    #         if isinstance(thrs, tuple):
    #             for key, values in eval_results_.items():
    #                 eval_results.update({
    #                     f'{key}_thr_{thr:.2f}': value.item()
    #                     for thr, value in zip(thrs, values)
    #                 })
    #         else:
    #             eval_results.update(
    #                 {k: v.item()
    #                  for k, v in eval_results_.items()})
    #
    #     if 'support' in metrics:
    #         # support_value = support(
    #         #     results, gt_labels, average_mode=average_mode)
    #         support_value = 0
    #         eval_results['support'] = support_value
    #
    #     precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
    #     if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
    #         # precision_recall_f1_values = precision_recall_f1(
    #         #     results, gt_labels, average_mode=average_mode, thrs=thrs)
    #         precision_recall_f1_values = [23,23]
    #         for key, values in zip(precision_recall_f1_keys,
    #                                precision_recall_f1_values):
    #             if key in metrics:
    #                 if isinstance(thrs, tuple):
    #                     eval_results.update({
    #                         f'{key}_thr_{thr:.2f}': value
    #                         for thr, value in zip(thrs, values)
    #                     })
    #                 else:
    #                     eval_results[key] = values
    #
    #     return eval_results
