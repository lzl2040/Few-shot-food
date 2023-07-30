# -*- encoding: utf-8 -*-
"""
File canp_head.py
Created on 2023/7/25 9:55
Copyright (c) 2023/7/25
@author: 
"""
import torch.nn as nn
from torch import Tensor
from typing import Dict, List
from model.base_head import BaseFewShotHead
import torch
import torch.nn.functional as F

from model.modules import LinearClassifierAdaptationNetwork
from util.tools import label_wrapper


class ConditionalAdaptiveNeuralProcessHead(BaseFewShotHead):
    def __init__(self, output_size):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d([1, 1])
        self.classifier_adaptation_network = LinearClassifierAdaptationNetwork(output_size)
        # used in meta testing
        self.support_feats = []
        self.support_labels = []
        self.class_support_feats_rep = None
        self.class_ids = None
        self.classifier_params = None

    def forward_train(self, support_feats: Tensor, support_labels: Tensor,
                      query_feats: Tensor, query_labels: Tensor,
                      **kwargs) -> Dict:
        """Forward training data.

        Args:
            support_feats (Tensor): Features of support data with shape (N, C).
            support_labels (Tensor): Labels of support data with shape (N).
            query_feats (Tensor): Features of query data with shape (N, C).
            query_labels (Tensor): Labels of query data with shape (N).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # 先avg
        support_feats = self.avg(support_feats).view(support_feats.shape[0], -1)
        query_feats = self.avg(query_feats).view(query_feats.shape[0], -1)

        class_ids = torch.unique(support_labels).cpu().tolist()
        query_labels = label_wrapper(query_labels, class_ids)
        # 获得特定类别的表示
        class_support_feats_rep = {}
        for class_id in class_ids:
            class_rep = support_feats[support_labels == class_id].mean(0, keepdim=True)
            class_support_feats_rep[class_id] = class_rep
        # 根据class support feats获得分类器的weight和bias
        classifier_params = self.classifier_adaptation_network(class_support_feats_rep)
        probs = F.linear(query_feats, classifier_params['weight_mean'], classifier_params['bias_mean'])
        # probs = F.softmax(probs,dim = 1)
        losses = self.loss(probs, query_labels)
        return losses

    def forward_support(self, x: Tensor, gt_label: Tensor, **kwargs) -> None:
        """Forward support data in meta testing."""
        self.support_feats.append(x)
        self.support_labels.append(gt_label)
        self.class_ids = None
        self.class_support_feats_rep = None
        self.classifier_params = None

    def forward_query(self, x: Tensor, **kwargs) -> List:
        """Forward query data in meta testing."""
        # print(x.shape)
        # print(self.mean_support_feats.shape)
        x = self.avg(x).view(x.shape[0], -1)
        probs = F.linear(x, self.classifier_params['weight_mean'], self.classifier_params['bias_mean'])
        pred = F.softmax(probs, dim = 1)
        pred = list(pred.detach().cpu().numpy())
        return pred

    def before_forward_support(self) -> None:
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        # reset saved features for testing new task
        self.support_feats.clear()
        self.support_labels.clear()
        self.class_ids = None
        self.class_support_feats_rep = None

    def before_forward_query(self) -> None:
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        support_feats = torch.cat(self.support_feats, dim=0)
        support_feats = self.avg(support_feats).view(support_feats.shape[0], -1)
        support_labels = torch.cat(self.support_labels, dim=0)
        self.class_ids, _ = torch.unique(support_labels).sort()
        self.class_support_feats_rep = {}
        for class_id in self.class_ids:
            class_rep = support_feats[support_labels == class_id].mean(0, keepdim=True)
            self.class_support_feats_rep[class_id.item()] = class_rep
        self.classifier_params = self.classifier_adaptation_network(self.class_support_feats_rep)