# -*- encoding: utf-8 -*-
"""
File anp_head.py
Created on 2023/7/16 22:11
Copyright (c) 2023/7/16
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
import torchvision.transforms as transforms

class AttentionNeuralProcessHead(BaseFewShotHead):
    def __init__(self, x_dim, x_trans_dim, r_dim, y_trans_dim, cross_out_dim, class_num, use_latent_path = False):
        super().__init__()
        self.class_num = class_num
        self.latent_path = use_latent_path
        # 空间维度上平均
        self.avg = nn.AdaptiveAvgPool2d([1, 1])
        # 将标签映射到和抽取的特征一样的空间
        self.proj_x = nn.Linear(in_features = x_dim, out_features = x_trans_dim,bias = False)
        self.map_label = nn.Linear(1, y_trans_dim, bias = False)
        self.deterministic_self_att = SelfAttention(x_dim = x_trans_dim + y_trans_dim, out_dim = r_dim,
                                                    n_attn_layers = 1)

        self.deterministic_cross_att = MultiheadAttender(kq_size = x_trans_dim, value_size = r_dim,
                                                         out_size = cross_out_dim)

        self.decoder = MLP(x1_dim = x_trans_dim, x2_dim = cross_out_dim, output_size=class_num,
                           n_hidden_layers=3, hidden_size=256, temper = 1, is_sum_merge=False,
                           need_softmax=True, need_hidden = True,is_output=True,dropout=0.3)
        # self.bn = nn.BatchNorm1d(num_features=50)
        self.support_feats_list = []
        self.support_feats = None
        self.support_labels = []
        self.class_ids = None
        self.deterministic_r = None
        self.latent_z = None
        # weights_init(self)

    def forward_train(self, support_feats, support_labels,
                      query_feats, query_labels,
                      **kwargs):
        class_ids = torch.unique(support_labels).cpu().tolist()
        query_labels = label_wrapper(query_labels, class_ids)
        Bs,C,H,W = support_feats.shape
        Bq,C,H,W = query_feats.shape
        # support:1 Bs C query:1 Bq C
        support_feats = self.avg(support_feats).view(Bs,-1).unsqueeze(0)
        support_feats = self.proj_x(support_feats)

        query_feats = self.avg(query_feats).view(Bq,-1).unsqueeze(0)
        # 1 Bq C_transf
        query_feats = self.proj_x(query_feats)

        support_labels = support_labels.unsqueeze(-1).float()
        # map label: 1 Bs C
        map_support_label = self.map_label(support_labels).unsqueeze(0)
        # 计算每个通道在N个样本上的均值和标准差
        mean = map_support_label.mean(dim = 1, keepdim=True)
        std = map_support_label.std(dim = 1, keepdim=True)

        # 进行Z-score归一化
        map_support_label = (map_support_label - mean) / std
        # 归一化
        # print(f"map:{map_support_label[0,0]}")
        # print(f"support:{support_feats[0,0]}")
        # combine feat: 1 Bs 2C
        combine_support_feats = torch.cat([support_feats, map_support_label], dim=-1)
        # deterministic feats: 1 Bs 256
        deterministic_support_r_feats = self.deterministic_self_att(combine_support_feats)
        # final r feats: 1 Bq 256
        # print(query_feats.shape)
        # print(support_feats.shape)
        # print("value:"+str(deterministic_support_r_feats.shape))
        final_r_feats = self.deterministic_cross_att(query_feats, support_feats, deterministic_support_r_feats)
        # print(f"query:{query_feats[0,0]}")
        # print(f"r feats:{final_r_feats[0,0]}")
        probs = self.decoder(query_feats, final_r_feats)
        # one hot编码
        # query_label_one_hot = F.one_hot(query_labels, num_classes=self.class_num)
        # print(probs.shape)
        # print(query_label_one_hot.shape)
        # print("probs:" + str(probs[0]))
        # print("label:"+str(query_label_one_hot[0]))
        losses = self.loss(probs,query_labels)
        return losses


    def forward_query(self, x, **kwargs):
        Bq, C, H, W = x.shape
        query_feats = self.avg(x).view(Bq, -1).unsqueeze(0)
        query_feats = self.proj_x(query_feats)
        # print(query_feats.shape)
        # print(self.support_feats.shape)
        # print(self.deterministic_r.shape)
        final_r_feats = self.deterministic_cross_att(query_feats, self.support_feats, self.deterministic_r)
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
        self.support_feats = self.proj_x(self.support_feats)
        support_labels = torch.cat(self.support_labels, dim=0)
        self.class_ids, _ = torch.unique(support_labels).sort()
        # 将label映射到跟feats一个空间
        support_labels = support_labels.unsqueeze(-1).float()
        map_support_label = self.map_label(support_labels).unsqueeze(0)
        # 计算每个通道在N个样本上的均值和标准差
        mean = map_support_label.mean(dim=1, keepdim=True)
        std = map_support_label.std(dim=1, keepdim=True)

        # 进行Z-score归一化
        map_support_label = (map_support_label - mean) / std
        # feats和label进行通道维度上拼接
        # print(map_support_label.shape)
        # print(self.support_feats.shape)
        combine_support_feats = torch.cat([self.support_feats, map_support_label], dim=-1)
        # 得到r feats
        self.deterministic_r = self.deterministic_self_att(combine_support_feats)

    def before_forward_support(self):
        self.support_feats_list.clear()
        self.support_labels.clear()
