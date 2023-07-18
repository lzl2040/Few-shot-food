# -*- encoding: utf-8 -*-
"""
File train.py
Created on 2023/7/7 16:37
Copyright (c) 2023/7/7
@author: 
"""
from mmcls.evaluation.metrics import Accuracy
from model.losses import CrossEntropyLoss
from model.meta_baseline_head import MetaBaselineHead
from model.meta_baseline import MetaBaseline

if __name__ == '__main__':
    # compute_accuracy = Accuracy(topk=1)
    # compute_loss = CrossEntropyLoss(loss_weight=1.0)
    # head = MetaBaselineHead()
    net = MetaBaseline(backbone = "resnet50", head = "meta_baseline")
    # for name,parameter in net.named_parameters():
    #     print(name,end=",")