# -*- encoding: utf-8 -*-
"""
File feature_reconstruction_network.py
Created on 2023/7/22 14:14
Copyright (c) 2023/7/22
@author: 
"""
from model.base_metric import BaseMetricClassifier
class FeatureReconstructionNetwork(BaseMetricClassifier):
    def __init__(self,
                 backbone,
                 head,
                 class_num):
        super().__init__(backbone = backbone, head = head,class_num = class_num)