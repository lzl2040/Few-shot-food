# -*- encoding: utf-8 -*-
"""
File attention_neural_process.py
Created on 2023/7/16 22:09
Copyright (c) 2023/7/16
@author: 
"""
from model.base_metric import BaseMetricClassifier
class AttentionNeuralProcess(BaseMetricClassifier):
    def __init__(self,
                 backbone,
                 head,
                 class_num):
        super().__init__(backbone = backbone, head = head,class_num = class_num)