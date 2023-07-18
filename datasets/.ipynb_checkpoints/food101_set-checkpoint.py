# -*- encoding: utf-8 -*-
"""
File food101_set.py
Created on 2023/7/7 11:56
Copyright (c) 2023/7/7
@author: 
"""
from datasets.base import BaseFewShotDataset
from typing_extensions import Literal
from typing import List, Optional, Sequence, Union
from util import tools
import os
import os.path as osp
import numpy as np

class Food101Dataset(BaseFewShotDataset):
    def __init__(self,
                 subset: Literal['train', 'test', 'val'] = 'train',
                 file_format: str = 'JPEG',
                 *args,
                 **kwargs):
        if isinstance(subset, str):
            subset = [subset]
        for subset_ in subset:
            assert subset_ in ['train', 'test', 'val']
        self.subset = subset
        self.file_format = file_format
        super().__init__(*args, **kwargs)

    def get_classes(
            self,
            classes: Optional[Union[Sequence[str],
                                    str]] = None) -> Sequence[str]:
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
        if classes is None:
            print('error')
        elif isinstance(classes, str):
            # take it as a file path
            class_names = tools.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def load_annotations(self) -> List:
        """Load annotation according to the classes subset."""
        img_file_list = {
            class_name: sorted(
                os.listdir(osp.join(self.data_prefix, 'images', class_name)),
                key=lambda x: int(x.split('.')[0]))
            for class_name in self.CLASSES
        }
        # print(img_file_list['apple_pie'])
        data_infos = []
        for subset_ in self.subset:
            ann_file = osp.join(self.data_prefix, 'meta', f'{subset_}.txt')
            assert osp.exists(ann_file), \
                f'Please download ann_file through {self.resource}.'
            with open(ann_file) as f:
                for i, line in enumerate(f):
                    # skip file head
                    # if i == 0:
                    #     continue
                    class_name, filename = line.strip().split('/')
                    # print(class_name,filename)
                    gt_label = self.class_to_idx[class_name]
                    info = {
                        'img_prefix':
                        osp.join(self.data_prefix, 'images', class_name),
                        'img_info': {
                            'filename': filename
                        },
                        'gt_label':
                        np.array(gt_label, dtype=np.int64)
                    }
                    data_infos.append(info)
        return data_infos
