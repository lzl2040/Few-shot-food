# -*- encoding: utf-8 -*-
"""
File meta_baseline.py
Created on 2023/7/7 16:37
Copyright (c) 2023/7/7
@author: 
"""
from model.base_metric import BaseMetricClassifier
class MetaBaseline(BaseMetricClassifier):
    """Implementation of `MetaBaseline <https://arxiv.org/abs/2003.04390>`_.

    Args:
        head (dict): Config of classification head for training.
    """

    def __init__(self,
                 backbone,
                 head,
                 neck = None):
        super().__init__(backbone = backbone, head = head)
