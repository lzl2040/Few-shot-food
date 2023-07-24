# -*- encoding: utf-8 -*-
"""
File debug_model.py
Created on 2023/7/22 8:39
Copyright (c) 2023/7/22
@author: 
"""
from model.base_metric import BaseMetricClassifier
class DebugModel(BaseMetricClassifier):
    def __init__(self,
                 backbone,
                 head,
                 class_num):
        super().__init__(backbone = backbone, head = head,class_num = class_num)