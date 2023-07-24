# -*- encoding: utf-8 -*-
"""
File conditional_neural_process.py
Created on 2023/7/21 10:16
Copyright (c) 2023/7/21
@author: 
"""
from model.base_metric import BaseMetricClassifier
class ConditionalNeuralProcess(BaseMetricClassifier):
    def __init__(self,
                 backbone,
                 head,
                 class_num):
        super().__init__(backbone = backbone, head = head,class_num = class_num)