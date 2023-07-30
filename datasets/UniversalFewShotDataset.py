# -*- encoding: utf-8 -*-
"""
File UniversalFewShotDataset.py
Created on 2023/7/30 12:33
Copyright (c) 2023/7/30
@author: 
"""
from datasets.base import BaseFewShotDataset
from typing_extensions import Literal
from typing import List, Optional, Sequence, Union
from util import tools
import os
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
class UniversalFewShotDataset(BaseFewShotDataset):
    def __init__(self,
                 img_size,
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
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        if subset[0] == 'train':
            pipeline = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(**norm_params)
        ])
        elif subset[0] == 'test':
            pipeline = transforms.Compose([
                transforms.Resize(size=int(img_size * 1.15)),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize(**norm_params)
            ])
        super().__init__(pipeline=pipeline, *args, **kwargs)

    def get_classes(
            self,
            classes: Optional[Union[Sequence[str], str]] = None) -> Sequence[str]:
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
        # print(img_file_list['apple_pie'])
        data_infos = []
        ann_file = self.ann_file
        with open(ann_file) as f:
            for i, line in enumerate(f):
                class_name, filename = line.strip().split('/')
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
