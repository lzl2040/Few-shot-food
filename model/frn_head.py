# -*- encoding: utf-8 -*-
"""
File frn_head.py
Created on 2023/7/22 14:15
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

def auxrank(support):
    way = support.size(0)
    shot = support.size(1)
    support = support/support.norm(2).unsqueeze(-1)
    L1 = torch.zeros((way**2-way)//2).long().cuda()
    L2 = torch.zeros((way**2-way)//2).long().cuda()
    counter = 0
    for i in range(way):
        for j in range(i):
            L1[counter] = i
            L2[counter] = j
            counter += 1
    s1 = support.index_select(0, L1) # (s^2-s)/2, s, d
    s2 = support.index_select(0, L2) # (s^2-s)/2, s, d
    dists = s1.matmul(s2.permute(0,2,1)) # (s^2-s)/2, s, s
    assert dists.size(-1)==shot
    frobs = dists.pow(2).sum(-1).sum(-1)
    return frobs.sum().mul(.03)

class FeatureReconstructionHead(BaseFewShotHead):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num

        self.r = nn.Parameter(torch.zeros(2), requires_grad = True).cuda()
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True).cuda()
        self.nll_loss = nn.NLLLoss()
        self.proj_img = nn.Linear(in_features=2048,out_features=512,bias=False)


        self.support_feats_list = []
        self.support_feats = None
        self.support_labels = []
        self.class_ids = None
        self.deterministic_r = None
        self.latent_z = None

    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):
        # query: way*query_shot*resolution, d
        # support: way, shot*resolution , d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1) / support.size(2)

        # correspond to lambda in the paper
        lam = reg * alpha.exp() + 1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0, 2, 1)  # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper

            sts = st.matmul(support)  # way, d, d
            m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()  # way, d, d
            hat = m_inv.matmul(sts)  # way, d, d

        else:
            # correspond to Equation 8 in the paper

            sst = support.matmul(st)  # way, shot*resolution, shot*resolution
            m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(
                lam)).inverse()  # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support)  # way, d, d
        # print("query shape:"+str(query.shape))
        # print("hat shape:"+str(hat.shape))
        # print("rho")
        Q_bar = query.matmul(hat).mul(rho)  # way, way*query_shot*resolution, d
        print("Q bar:"+str(Q_bar.shape))
        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)  # way*query_shot*resolution, way

        return dist

    def forward_train(self, support_feats, support_labels,
                      query_feats, query_labels,
                      **kwargs):
        class_ids = torch.unique(support_labels).cpu().tolist()
        query_labels = label_wrapper(query_labels, class_ids)
        Bs,C,H,W = support_feats.shape
        Bq,C,H,W = query_feats.shape
        support_shots = Bs // self.class_num
        query_shots = Bq // self.class_num
        resolution = H * W
        # print("reso:"+str(resolution))
        # print("query shot:"+str(query_shots))
        target = torch.LongTensor([i // query_shots for i in range(query_shots * self.class_num)]).cuda()
        # transform shape
        ## support:way, shot*resolution , d
        support_feats = support_feats.view(Bs,C,-1).permute(0,2,1)\
            .contiguous().view(self.class_num, support_shots * resolution, C)
        query_feats = query_feats.view(Bq, C, -1).permute(0,2,1)\
            .contiguous().view(self.class_num * query_shots * resolution, C)

        recon_dist = self.get_recon_dist(query=query_feats, support=support_feats, alpha=self.r[0],
                                         beta=self.r[1])  # way*query_shot*resolution, way
        neg_l2_dist = recon_dist.neg().view(self.class_num * query_shots, resolution, self.class_num).mean(1)

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)
        frn_loss = self.nll_loss(log_prediction,target)
        aux_loss = auxrank(support_feats)
        loss = frn_loss + aux_loss
        losses = {}
        losses['loss'] = loss
        return losses


    def forward_query(self, x, **kwargs):
        Bq, C, H, W = x.shape
        resolution = H * W
        query_shots = Bq // len(self.class_ids)
        query_feats = x.view(Bq, C, -1).permute(0, 2, 1) \
            .contiguous().view(len(self.class_ids) * query_shots * resolution, C)
        recon_dist = self.get_recon_dist(query = query_feats, support = self.support_feats, alpha = self.r[0],
                                         beta = self.r[1])  # way*query_shot*resolution, way
        neg_l2_dist = recon_dist.neg().view(len(self.class_ids) * query_shots, resolution, len(self.class_ids)).mean(1)

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)
        return log_prediction

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
        self.support_feats = self.support_feats.view(Bs, C, -1).permute(0, 2, 1) \
            .contiguous().view(len(self.class_ids), support_shots * resolution, C)

    def before_forward_support(self):
        self.support_feats_list.clear()
        self.support_labels.clear()
        self.class_ids = None
