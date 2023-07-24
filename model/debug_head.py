# -*- encoding: utf-8 -*-
"""
File debug_head.py
Created on 2023/7/22 8:40
Copyright (c) 2023/7/22
@author: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_head import BaseFewShotHead
from model.modules import *
from util.tools import label_wrapper
from model.losses import *
from torch.nn import NLLLoss

class DebugHead(BaseFewShotHead):
    def __init__(self, x_dim, x_trans_dim, y_trans_dim, class_num):
        super().__init__()
        self.class_num = class_num
        # 空间维度上平均
        self.avg = nn.AdaptiveAvgPool2d([1, 1])
        # 将标签映射到和抽取的特征一样的空间
        self.proj_x = nn.Linear(in_features = x_dim, out_features = x_trans_dim,bias = False)
        self.aggregate = MultiheadAttender(input_kq_size=x_dim, kq_size=x_trans_dim, value_size=x_dim,
                                           n_heads=8, out_size=x_dim, is_post_process=False)

        self.output_layer = nn.Linear(x_dim,class_num,bias = False)
        self.x_dim = x_dim
        self.nll_loss = NLLLoss()
        self.support_feats_list = []
        self.support_feats = None
        self.support_labels = []
        self.class_ids = None

    def forward_train(self, support_feats, support_labels,
                      query_feats, query_labels,
                      **kwargs):
        class_ids = torch.unique(support_labels).cpu().tolist()
        query_labels = label_wrapper(query_labels, class_ids)
        Bs,C,H,W = support_feats.shape
        Bq,C,H,W = query_feats.shape
        resolution = H * W
        query_shots = Bq // self.class_num
        support_shots = Bs // self.class_num
        # transform shape
        ## support: num_ways support_shot * resolution d
        ## query: num_ways query_shot * resolution d
        support_feats = support_feats.view(Bs,C,-1).permute(0,2,1) \
            .contiguous().view(self.class_num, support_shots * resolution, C)
        query_feats = query_feats.view(Bq,C,-1).permute(0,2,1) \
            .contiguous().view(self.class_num, query_shots * resolution, C)
        # aggregate feats: num_ways query_shot * resolution dv
        # print(support_feats.shape)
        # print(query_feats.shape)
        aggregate_feats = self.aggregate(query_feats,support_feats,support_feats)
        # print("agg:"+str(aggregate_feats.shape))
        aggregate_feats = aggregate_feats.view(self.class_num * query_shots, resolution, self.x_dim)
        mean_feats = torch.mean(aggregate_feats,dim=1)
        probs = self.output_layer(mean_feats)
        # one hot编码
        query_label_one_hot = F.one_hot(query_labels, num_classes=self.class_num)
        losses = {}
        loss = cross_entropy(probs, query_label_one_hot.float())
        losses['loss'] = loss
        return losses

    def forward_query(self, x, **kwargs):
        Bq, C, H, W = x.shape
        resolution = H * W
        query_shots = Bq // len(self.class_ids)
        query_feats = x.view(Bq, C, -1).permute(0, 2, 1) \
            .contiguous().view(len(self.class_ids), query_shots * resolution, C)
        # print(query_feats.shape)
        # print(self.support_feats.shape)
        aggregate_feats = self.aggregate(query_feats, self.support_feats, self.support_feats)
        # print("agg:"+str(aggregate_feats.shape))
        aggregate_feats = aggregate_feats.view(len(self.class_ids) * query_shots, resolution, -1)
        mean_feats = torch.mean(aggregate_feats, dim=1)
        probs = self.output_layer(mean_feats)
        probs = torch.softmax(probs,dim = -1)
        probs = list(probs.detach().cpu().numpy())
        return probs

    def forward_support(self, x, gt_label, **kwargs):
        self.support_feats_list.append(x)
        self.support_labels.append(gt_label)

    def before_forward_query(self):
        self.support_feats = torch.cat(self.support_feats_list, dim=0)
        Bs, C, H, W = self.support_feats.shape
        resolution = H * W
        support_labels = torch.cat(self.support_labels, dim=0)
        self.class_ids = torch.unique(support_labels).cpu().tolist()
        support_shots = Bs // len(self.class_ids)
        self.support_feats = self.support_feats.view(Bs,C,-1).permute(0,2,1) \
            .contiguous().view(len(self.class_ids), support_shots * resolution, C)

    def before_forward_support(self):
        self.support_feats_list.clear()
        self.support_labels.clear()
        self.class_ids = None
