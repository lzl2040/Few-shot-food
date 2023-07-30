# -*- encoding: utf-8 -*-
"""
File canp.py
Created on 2023/7/25 9:55
Copyright (c) 2023/7/25
@author: 
"""
from model.base_metric import BaseMetricClassifier
class ConditionalAdaptiveNeuralProcess(BaseMetricClassifier):
    def __init__(self,
                 backbone,
                 head,
                 class_num):
        super().__init__(backbone = backbone, head = head,class_num = class_num)