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


class SelfAttention(nn.Module):
    def __init__(self, input_dim, k_dim,v_dim, head = 1, dropout = 0):
        super().__init__()
        self.k_dim = k_dim
        self.head = head
        self.linear_q = nn.Linear(input_dim // head, k_dim // head, bias=False)
        self.linear_k = nn.Linear(input_dim // head, k_dim // head, bias=False)
        self.linear_v = nn.Linear(input_dim // head, v_dim // head, bias=False)
        self.dropout = nn.Dropout(p = dropout) if dropout > 0 else nn.Identity()

    def forward(self,x):
        # x dim: B N C
        B,N,C = x.shape
        if self.head != 1:
            x = x.view(B, N, self.head, C // self.head)
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        affinity_matrix = torch.matmul(q, k.transpose(-1,-2)) / self.k_dim * self.head
        attention = torch.softmax(affinity_matrix,dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention,v)
        if self.head != 1:
            context = context.view(B, N, -1)
        return context

class CrosssAttention(nn.Module):
    def __init__(self, k_input_dim, v_input_dim, k_dim,v_dim, head = 1, dropout = 0):
        super().__init__()
        self.k_dim = k_dim
        self.head = head
        self.linear_q = nn.Linear(k_input_dim // head, k_dim // head,bias=False)
        self.linear_k = nn.Linear(k_input_dim // head, k_dim // head,bias=False)
        # self.linear_v = nn.Linear(v_input_dim // head, v_dim // head,bias=False)
        self.dropout = nn.Dropout(p = dropout) if dropout > 0 else nn.Identity()

    def forward(self,q_x, k_x, v_x):
        # x dim: B N C
        B,N,Cq = q_x.shape
        _,_,Cv = v_x.shape
        if self.head != 1:
            q_x = q_x.view(B, N, self.head, Cq // self.head)
            k_x = k_x.view(B, N, self.head, Cq // self.head)
            v_x = v_x.view(B, N, self.head, Cv // self.head)
            # print(v_x.shape)
        q = self.linear_q(q_x)
        k = self.linear_k(k_x)
        # v = self.linear_v(v_x)
        # B * N * 8 * 512 / 8
        v = v_x
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)

        affinity_matrix = torch.matmul(q, k.transpose(-1,-2)) / self.k_dim * self.head
        attention = torch.softmax(affinity_matrix,dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention,v)
        if self.head != 1:
            context = context.view(B, N, -1)
        return context


class MLP(nn.Module):
    def __init__(self,r_dim, z_dim, q_dim, class_num):
        super().__init__()
        # self.proj_rz = nn.Linear(r_dim + z_dim, q_dim, bias = False)
        all_dim = q_dim + r_dim
        self.class_num = class_num
        self.linear1 = nn.Linear(in_features=all_dim, out_features=512,bias=False)
        self.linear2 = nn.Linear(in_features=512, out_features = 256,bias=False)
        self.linear3 = nn.Linear(in_features=256,out_features=32,bias=False)
        self.output_layer = nn.Linear(in_features=32 * 14 * 14, out_features = class_num)

    def forward(self, query, r, z = None):
        if z:
            expand_z = z.expand(-1, r.shape[1], -1)
            rz_merge = torch.cat([expand_z, r],dim = -1)
            rz_merge = self.proj_rz(rz_merge)
        else:
            rz_merge = r
        # combine_feats:B N(196) C(1024+512)
        combine_feats = torch.cat([rz_merge, query],dim = -1)
        linear1_feats = torch.sigmoid(self.linear1(combine_feats))
        linear2_feats = torch.sigmoid(self.linear2(linear1_feats))
        linear3_feats = torch.sigmoid(self.linear3(linear2_feats))
        B,N,C = linear3_feats.shape
        final_feats = linear3_feats.view(B,-1)
        prob = torch.softmax(self.output_layer(final_feats),dim = -1)
        # print(prob)
        return prob



