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
from model.modules import SelfAttention, CrosssAttention, MLP
from util.tools import label_wrapper
from model.losses import *


class AttentionNeuralProcessHead(BaseFewShotHead):
    def __init__(self, self_input_dim, cross_input_dim, q_dim, v_dim, spatial_size, class_num,
                 head = 8, dropout = 0.2, use_latent_path = False):
        super().__init__()
        self.class_num = class_num
        self.latent_path = use_latent_path
        # 空间维度上平均
        self.avg = nn.AdaptiveAvgPool2d([1, 1])
        # 将标签映射到和抽取的特征一样的空间
        # self.map_label = nn.Linear(1, spatial_size, bias = False)
        self.map_label = nn.Linear(1, 2048, bias=False)
        # 转换[x,y]维度
        # self.proj_xy = nn.Linear(self_input_dim + 1, self_input_dim, bias = False)
        # anp
        self.deterministic_self_att = SelfAttention(input_dim = self_input_dim, k_dim = q_dim, v_dim = v_dim,
                                                    head = head,dropout=dropout)
        # self.proj_support_feats = nn.Linear(in_features=self_input_dim, out_features = q_dim)
        self.deterministic_cross_att = CrosssAttention(k_input_dim=cross_input_dim, v_input_dim = v_dim,
                                                       k_dim=q_dim, v_dim=v_dim, head=head,dropout=dropout)
        if self.latent_path:
            self.latent_self_att = SelfAttention(input_dim = self_input_dim, k_dim=q_dim, v_dim = v_dim, head = head)
        self.decoder = MLP(r_dim = v_dim, z_dim=v_dim, q_dim = cross_input_dim, class_num=class_num)

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
        support_feats = self.avg(support_feats).view(Bs,-1).unsqueeze(0)
        query_feats = self.avg(query_feats).view(Bq,-1).unsqueeze(0)
        support_labels = support_labels.unsqueeze(-1).float()
        # map label: 1 Bs 1024
        map_support_label = self.map_label(support_labels).unsqueeze(0)
        # map_support_label = support_labels.expand(-1, 2048).unsqueeze(0)
        # print(map_support_label)
        # combine feat: 1 Bs 2048
        combine_support_feats = torch.cat([support_feats, map_support_label], dim=-1)
        # print(combine_support_feats.shape)
        # deterministic feats: 1 Bs 256
        # deterministic_support_r_feats = self.proj_support_feats(combine_support_feats)
        deterministic_support_r_feats = self.deterministic_self_att(combine_support_feats)
        if self.latent_path:
            # latent feats: 1 Bs 256
            latent_support_z_feats = self.latent_self_att(support_feats)
            # latent_support_z_feats = torch.mean(latent_support_z_feats, dim=1, keepdim=True)
        # final r feats: 1 Bq 256
        final_r_feats = self.deterministic_cross_att(query_feats, support_feats, deterministic_support_r_feats)
        if self.latent_path:
            probs = self.decoder(query_feats, final_r_feats, latent_support_z_feats)
        else:
            probs = self.decoder(query_feats, final_r_feats)
        # one hot编码
        query_label_one_hot = F.one_hot(query_labels, num_classes=self.class_num)
        # print(probs[1:3])
        # print(query_label_one_hot[1:3])

        # # support_labels shape: B
        # # 转换为B N(14*14) C(1024)
        # support_feats = support_feats.view(Bs, C, -1).transpose(1, 2)
        # query_feats = query_feats.view(Bq, C, -1).transpose(1, 2)
        # # 维度变化： B * class_num * 1 -> B * class_num * q_dim
        # support_labels = support_labels.unsqueeze(-1).float()
        # map_support_label = self.map_label(support_labels).unsqueeze(-1).expand(-1, -1, C)
        # # 拼接
        # # one-hot编码和support feats进行通道维度拼接 B N 2048
        # combine_support_feats = torch.cat([support_feats,map_support_label],dim = -1)
        # # combine_support_feats = self.proj_xy(combine_support_feats)
        # # 输入到determinstic self-attention里面去,得到r
        # # 维度变成BNC
        # deterministic_support_r_feats = self.deterministic_self_att(combine_support_feats)
        # # 将support features输入到latent self-attention里面去,得到s，然后取平均得到latent variable z
        # # 维度变成B N C
        # if self.latent_path:
        #     latent_support_z_feats = self.latent_self_att(support_feats)
        #     latent_support_z_feats = torch.mean(latent_support_z_feats, dim = 1, keepdim = True)
        # # 将r,support feats和query feats输入到deterministic cross-attention里面，得到最终的r
        # final_r_feats = self.deterministic_cross_att(query_feats,support_feats,deterministic_support_r_feats)
        # # 将z,r和query feats输入到decoder中
        # if self.latent_path:
        #     probs = self.decoder(query_feats,final_r_feats, latent_support_z_feats)
        # else:
        #     probs = self.decoder(query_feats, final_r_feats)
        # # query label的one hot编码
        # query_label_one_hot = F.one_hot(query_labels,num_classes = self.class_num)
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
