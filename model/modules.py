# -*- encoding: utf-8 -*-
"""
File modules.py
Created on 2023/7/17 10:59
Copyright (c) 2023/7/17
@author: 
"""
import math
import torch
import torch.nn as nn
from util.tools import weights_init
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, input_dim, k_dim,v_dim, head = 1, dropout = 0):
        super().__init__()
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.head = head
        self.linear_q = nn.Linear(input_dim, k_dim, bias=False)
        self.linear_k = nn.Linear(input_dim, k_dim, bias=False)
        self.linear_v = nn.Linear(input_dim, v_dim, bias=False)
        self.output_linear = nn.Linear(v_dim, v_dim)
        self.dropout = nn.Dropout(p = dropout) if dropout > 0 else nn.Identity()
        self.init_weight()
    def init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                # nn.init.constant_(m.bias, 0)
    def forward(self,x):
        # x dim: B N C
        _,N,C = x.shape
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        if self.head != 1:
            q = q.view(1, N, self.head, self.k_dim // self.head).transpose(1, 2)
            k = k.view(1, N, self.head, self.k_dim // self.head).transpose(1, 2)
            v = v.view(1, N, self.head, self.v_dim // self.head).transpose(1, 2)
        affinity_matrix = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.k_dim // self.head)
        attention = torch.softmax(affinity_matrix,dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention,v)
        if self.head != 1:
            context = context.transpose(1,2).contiguous().view(1,N,-1)
        attention_out = self.output_linear(context)
        return attention_out

class CrosssAttention(nn.Module):
    def __init__(self, k_input_dim, v_input_dim, k_dim,v_dim, head = 1, dropout = 0):
        super().__init__()
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.head = head
        self.linear_q = nn.Linear(k_input_dim, k_dim, bias=False)
        self.linear_k = nn.Linear(k_input_dim, k_dim, bias=False)
        self.linear_v = nn.Linear(v_input_dim, v_dim, bias=False)
        self.output_linear = nn.Linear(v_dim, v_dim)
        self.dropout = nn.Dropout(p = dropout) if dropout > 0 else nn.Identity()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                # nn.init.constant_(m.bias, 0)

    def forward(self,q_x, k_x, v_x):
        # x dim: B N C
        _,Bq,Cq = q_x.shape
        _,Bs,Cv = v_x.shape
        q = self.linear_q(q_x)
        k = self.linear_k(k_x)
        # v = self.linear_v(v_x)
        # B * N * 8 * 512 / 8
        v = v_x
        if self.head != 1:
            q = q.view(1, Bq, self.head, self.k_dim // self.head).transpose(1,2)
            k = k.view(1, Bs, self.head, self.k_dim // self.head).transpose(1,2)
            v = v.view(1, Bs, self.head, self.v_dim // self.head).transpose(1,2)
            # print(v_x.shape)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)

        affinity_matrix = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.k_dim // self.head)
        attention = torch.softmax(affinity_matrix,dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention,v)
        if self.head != 1:
            context = context.transpose(1,2).contiguous().view(1,Bq,-1)
        attention_out = self.output_linear(context)
        return context


class MLP(nn.Module):
    def __init__(self,r_dim, z_dim, q_dim, class_num, temper = 0.1):
        super().__init__()
        # self.proj_rz = nn.Linear(r_dim + z_dim, q_dim, bias = False)
        all_dim = q_dim + r_dim
        self.class_num = class_num
        self.linear1 = nn.Linear(in_features=all_dim, out_features=512,bias = False)
        self.linear2 = nn.Linear(in_features=512, out_features = 64,bias = False)
        self.output_layer = nn.Linear(in_features=64,out_features=self.class_num,bias = False)
        self.temper = temper
        self.init_weight()
    def init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                # nn.init.constant_(m.bias, 0)

    def forward(self, query, r, z = None):
        # query :1 Bq 1024 r:1 Bq 256
        if z:
            expand_z = z.expand(-1, r.shape[1], -1)
            rz_merge = torch.cat([expand_z, r],dim = -1)
            rz_merge = self.proj_rz(rz_merge)
        else:
            rz_merge = r
        # combine_feats:1 Bq 1280
        combine_feats = torch.cat([rz_merge, query],dim = -1)
        linear1_feats = torch.sigmoid(self.linear1(combine_feats))
        linear2_feats = torch.sigmoid(self.linear2(linear1_feats))
        prob = self.output_layer(linear2_feats).squeeze(0)
        # 添加温度系数
        prob = torch.softmax(prob / self.temper,dim=-1)
        return prob



