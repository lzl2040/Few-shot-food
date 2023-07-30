# -*- encoding: utf-8 -*-
"""
File cnp_head.py
Created on 2023/7/21 10:17
Copyright (c) 2023/7/21
@author: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_head import BaseFewShotHead
from model.modules import *
from util.tools import label_wrapper
from model.losses import *


class ConditionalNeuralProcessHead(BaseFewShotHead):
    def __init__(self, x_dim, x_trans_dim, r_dim, y_trans_dim, class_num):
        super().__init__()
        self.class_num = class_num
        # 空间维度上平均
        self.avg = nn.AdaptiveAvgPool2d([1, 1])
        # 将标签映射到和抽取的特征一样的空间
        self.proj_x = nn.Linear(in_features = x_dim, out_features = x_trans_dim,bias = False)
        self.map_label = nn.Linear(1, y_trans_dim, bias = False)
        self.XYEncoder = MLP(x1_dim=x_trans_dim,x2_dim=y_trans_dim,hidden_size=r_dim,output_size=r_dim,
                             n_hidden_layers=2,dropout=0.5)
        self.decoder = MLP(x1_dim = x_trans_dim, x2_dim=r_dim,hidden_size=256,output_size=class_num,
                           n_hidden_layers=2, need_softmax = True, is_output=True,dropout=0.5)
        self.support_feats_list = []
        self.support_feats = None
        self.support_labels = []
        self.class_ids = None
        self.deterministic_r = None
        self.latent_z = None

    def forward_train(self, support_feats, support_labels,
                      query_feats, query_labels,
                      **kwargs):
        class_ids = torch.unique(support_labels).cpu().tolist()
        query_labels = label_wrapper(query_labels, class_ids)
        Bs,C,H,W = support_feats.shape
        Bq,C,H,W = query_feats.shape
        # support:1 Bs C query:1 Bq C
        support_feats = self.avg(support_feats).view(Bs, -1)
        mean_support_feats = torch.cat([
            support_feats[support_labels == class_id].mean(0, keepdim=True)
            for class_id in class_ids
        ], dim = 0)
        mean_support_feats = mean_support_feats.unsqueeze(0)
        # 1 Bs C_transf
        # print(mean_support_feats.shape)
        support_feats = self.proj_x(mean_support_feats)

        query_feats = self.avg(query_feats).view(Bq,-1).unsqueeze(0)
        # 1 Bq C_transf
        query_feats = self.proj_x(query_feats)
        support_labels = torch.unique(support_labels).unsqueeze(-1).float()
        # map label: 1 Bs C
        map_support_label = self.map_label(support_labels).unsqueeze(0)

        # 计算每个通道在N个样本上的均值和标准差
        mean = map_support_label.mean(dim=1, keepdim=True)
        std = map_support_label.std(dim=1, keepdim=True)
        # 进行Z-score归一化
        map_support_label = (map_support_label - mean) / std
        # combine feat: 1 Bs 2C
        mid_rep = self.XYEncoder(support_feats,map_support_label)
        # print(mid_rep.shape)
        representation = torch.mean(mid_rep,dim = 1,keepdim=True).expand(-1,Bq,-1)
        # print(representation.shape)
        probs = self.decoder(query_feats, representation)
        # print(probs.shape)
        # one hot编码
        # print(probs.shape)
        query_label_one_hot = F.one_hot(query_labels, num_classes=self.class_num)
        # print("probs:" + str(probs[0]))
        # print("label:" + str(query_label_one_hot[0]))
        losses = {}
        loss = cross_entropy(probs,query_label_one_hot.float())
        losses['loss'] = loss
        return losses


    def forward_query(self, x, **kwargs):
        Bq, C, H, W = x.shape
        query_feats = self.avg(x).view(Bq, -1).unsqueeze(0)
        # print(query_feats.shape)
        # print(self.support_feats.shape)
        # print(self.deterministic_r.shape)
        final_r_feats = self.deterministic_cross_att(query_feats, self.support_feats, self.deterministic_r)
        if self.latent_path:
            probs = self.decoder(query_feats,final_r_feats, self.latent_z)
        else:
            probs = self.decoder(query_feats, final_r_feats)
        probs = list(probs.detach().cpu().numpy())
        return probs

    def forward_support(self, x, gt_label, **kwargs):
        self.support_feats_list.append(x)
        self.support_labels.append(gt_label)

    def before_forward_query(self):
        self.support_feats = torch.cat(self.support_feats_list, dim=0)
        Bs, C, H, W = self.support_feats.shape
        self.support_feats = self.avg(self.support_feats).view(Bs, -1).unsqueeze(0)
        support_labels = torch.cat(self.support_labels, dim=0)
        self.class_ids, _ = torch.unique(support_labels).sort()
        # 将label映射到跟feats一个空间
        support_labels = support_labels.unsqueeze(-1).float()
        map_support_label = self.map_label(support_labels).unsqueeze(0)
        # feats和label进行通道维度上拼接
        combine_support_feats = torch.cat([self.support_feats, map_support_label], dim=-1)
        # 得到r feats
        self.deterministic_r = self.deterministic_self_att(combine_support_feats)

    def before_forward_support(self):
        self.support_feats_list.clear()
        self.support_labels.clear()
