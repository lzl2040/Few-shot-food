# -*- encoding: utf-8 -*-
"""
File modules.py
Created on 2023/7/17 10:59
Copyright (c) 2023/7/17
@author: 
"""
import math
import abc
import torch
import torch.nn as nn
from util.tools import weights_init, linear_init
import torch.nn.functional as F
import numpy as np

class BaseAttender(abc.ABC, nn.Module):
    """
    Base Attender module.
    """

    def __init__(self, kq_size, value_size, out_size, is_normalize=True, dropout=0):
        super().__init__()
        self.kq_size = kq_size
        self.value_size = value_size
        self.out_size = out_size
        self.is_normalize = is_normalize
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.is_resize = self.value_size != self.out_size

        if self.is_resize:
            self.resizer = nn.Linear(self.value_size, self.out_size)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, queries, keys, values, **kwargs):
        """
        Compute the attention between given key and queries.

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries : torch.Tensor, size=[batch_size, n_queries, kq_size]
        values : torch.Tensor, size=[batch_size, n_keys, value_size]

        Return
        ------
        context : torch.Tensor, size=[batch_size, n_queries, out_size]
        """
        logits = self.score(queries, keys, **kwargs)

        attn = self.logits_to_attn(logits)

        attn = self.dropout(attn)

        # attn : size=[batch_size, n_queries, n_keys]
        # values : size=[batch_size, n_keys, value_size]
        # print(attn.shape)
        # print(values.shape)
        # print("att:"+str(attn.shape))
        # print("value:"+str(values.shape))
        context = torch.matmul(attn, values)

        if self.is_resize:
            context = self.resizer(context)

        return context

    def logits_to_attn(self, logits):
        """Convert logits to attention."""
        if self.is_normalize:
            attn = logits.softmax(dim=-1)
        else:
            attn = logits
        return attn

    @abc.abstractmethod
    def score(keys, queries, **kwargs):
        """Score function which returns the logits between keys and queries."""
        pass


class DotAttender(BaseAttender):
    """
    Dot product attention.
    """

    def __init__(self, *args, is_scale=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_scale = is_scale

    def score(self, queries, keys):
        # b: batch_size, q: n_queries, k: n_keys, d: kq_size
        # e.g. if keys have 4 dimension it means that different queries will
        # be associated with different keys
        # keys_shape = "bqkd" if len(keys.shape) == 4 else "bkd"
        # queries_shape = "bqxd" if len(queries.shape) == 4 else "bqd"
        #
        # # [batch_size, n_queries, kq_size]
        # logits = torch.einsum(
        #     "{},{}->bqxk".format(keys_shape, queries_shape), keys, queries
        # )
        # print("query:"+str(queries.shape))
        # print("keys:"+str(keys.shape))
        logits = torch.matmul(queries,keys.transpose(-1,-2))
        # print("logits:"+str(logits.shape))

        if self.is_scale:
            kq_size = queries.size(-1)
            logits = logits / math.sqrt(kq_size)

        return logits

class MultiheadAttender(nn.Module):
    """
    Multihead attention mechanism.
    """

    def __init__(
        self,
        kq_size,
        value_size,
        out_size,
        input_kq_size = None,
        n_heads=8,
        is_post_process=True,
        dropout=0,
    ):
        super().__init__()
        # only 3 transforms for scalability but actually as if using n_heads * 3 layers
        if input_kq_size == None:
            input_kq_size = kq_size
        self.key_transform = nn.Linear(input_kq_size, kq_size, bias=False)
        self.query_transform = nn.Linear(input_kq_size, kq_size,bias = False)
        self.value_transform = nn.Linear(value_size, value_size, bias=False)
        self.dot = DotAttender(kq_size, value_size, out_size, is_scale=True, dropout=dropout)
        self.n_heads = n_heads
        self.kq_head_size = kq_size // self.n_heads
        self.value_head_size = value_size // self.n_heads
        self.kq_size = kq_size
        self.value_size = value_size
        self.out_size = out_size
        self.post_processor = (
            nn.Linear(value_size, out_size)
            if is_post_process or value_size != out_size
            else None
        )

        assert kq_size % n_heads == 0, "{} % {} != 0".format(kq_size, n_heads)
        assert value_size % n_heads == 0, "{} % {} != 0".format(value_size, n_heads)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)
        # change initialization because real output is not kqv_size but head_size
        # just coded so for convenience and scalability
        std = math.sqrt(2.0 / (self.kq_size + self.kq_head_size))
        nn.init.normal_(self.key_transform.weight, mean=0, std=std)
        nn.init.normal_(self.query_transform.weight, mean=0, std=std)
        std = math.sqrt(2.0 / (self.value_size + self.value_head_size))
        nn.init.normal_(self.value_transform.weight, mean=0, std=std)

    def forward(self, queries, keys, values, **kwargs):
        """
        Compute the attention between given key and queries.

        Parameters
        ----------
        keys: torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries: torch.Tensor, size=[batch_size, n_queries, kq_size]
        values: torch.Tensor, size=[batch_size, n_keys, value_size]
        rel_pos_enc: torch.Tensor, size=[batch_size, n_queries, n_keys, kq_size]
            Positional encoding with the differences between every key and query.

        Return
        ------
        context : torch.Tensor, size=[batch_size, n_queries, out_size]
        """
        # print("enter")
        # print(values.shape)
        # print("input values:"+str(values.shape))
        queries = self.query_transform(queries)
        keys = self.key_transform(keys)
        values = self.value_transform(values)
        # print("queries transform:"+str(queries.shape))
        # print("values transform:"+str(values.shape))

        # Make multihead. Size = [batch_size * n_heads, {n_keys, n_queries}, head_size]
        queries = self._make_multiheaded(queries, self.kq_head_size)
        # keys have to add relative position before splitting head
        keys = self._make_multiheaded(keys, self.kq_head_size)
        # print("pre")
        # print(values.shape)
        values = self._make_multiheaded(values, self.value_head_size)
        # print("multi head:"+str(values.shape))
        # print(queries.shape)
        # print(keys.shape)
        # print(values.shape)

        # Size = [batch_size * n_heads, n_queries, head_size]
        context = self.dot(queries, keys, values)
        # print("context shape:"+str(context.shape))

        # Size = [batch_size, n_queries, value_size]
        context = self._concatenate_multiheads(context, self.value_head_size)

        if self.post_processor is not None:
            context = self.post_processor(context)

        return context

    def _make_multiheaded(self, kvq, head_size):
        """Make a key, value, query multiheaded by stacking the heads as new batches."""
        batch_size = kvq.size(0)
        kvq = kvq.view(batch_size, -1, self.n_heads, head_size)
        kvq = kvq.permute(0, 2, 1, 3).contiguous()
        return kvq

    def _concatenate_multiheads(self, kvq, head_size):
        """Reverts `_make_multiheaded` by concatenating the heads."""
        batch_size = kvq.shape[0]
        kvq = kvq.view(batch_size, self.n_heads, -1, head_size)
        kvq = kvq.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.n_heads * head_size)
        return kvq

class SelfAttention(nn.Module):
    """Self Attention Layer.
    """
    def __init__(
        self,
        x_dim,
        out_dim=None,
        n_attn_layers=2,
        **kwargs
    ):
        super().__init__()

        self.attn_layers = nn.ModuleList(
            [
                MultiheadAttender(x_dim, x_dim, x_dim, **kwargs)
                for _ in range(n_attn_layers)
            ]
        )

        self.is_resize = out_dim is not None
        if self.is_resize:
            self.resize = nn.Linear(x_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
        add_to_keys = 0
        out = X

        for attn_layer in self.attn_layers:
            out = attn_layer(out + add_to_keys, out, out)
            # print("out shape:"+str(out.shape))

        if self.is_resize:
            out = self.resize(out)

        return out

class MLP(nn.Module):
    """General MLP class.
    is_res : bool, optional
        Whether to use residual connections.
    """

    def __init__(
        self,
        x1_dim,
        x2_dim,
        output_size,
        hidden_size=32,
        temper = 0.01,
        n_hidden_layers=1,
        activation=nn.ReLU(),
        is_bias=False,
        dropout=0,
        is_res=True,
        is_sum_merge = True,
        need_softmax = False,
        need_hidden = False,
    ):
        super().__init__()
        if is_sum_merge:
            self.input_size = x1_dim
        else:
            self.input_size = x1_dim + x2_dim
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.need_hidden = need_hidden
        self.n_hidden_layers = n_hidden_layers
        self.is_res = is_res
        self.temper = temper
        if self.need_hidden:
            self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=is_bias)
            self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
            self.activation = activation
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size, bias=is_bias)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )
        if self.need_hidden:
            self.out = nn.Linear(self.hidden_size, self.output_size, bias=is_bias)
        else:
            self.out = nn.Linear(self.input_size, self.output_size, bias=is_bias)
        self.need_reshape = False
        self.is_sum_merge = is_sum_merge
        self.need_softmax = need_softmax
        if x1_dim != x2_dim and is_sum_merge:
            self.need_reshape = True
            self.resizer = nn.Linear(x2_dim,x1_dim)

        self.reset_parameters()

    def forward(self, query_feats, rep):
        if self.is_sum_merge:
            resize_rep = rep
            if self.need_reshape:
                resize_rep = self.resizer(rep)
            sum_x = query_feats + resize_rep
            x = torch.relu(sum_x)
        else:
            x = torch.cat([query_feats, rep],dim = -1)
        if self.need_hidden:
            out = self.to_hidden(x)
            out = self.activation(out)
            x = self.dropout(out)
        for linear in self.linears:
            out = linear(x)
            out = self.activation(out)
            if self.is_res:
                out = out + x
            out = self.dropout(out)
            x = out
        out = self.out(x)
        out = out.squeeze(0)
        if self.need_softmax:
            out = torch.softmax(out / self.temper,dim = -1)
        return out

    def reset_parameters(self):
        if self.need_hidden:
            linear_init(self.to_hidden, activation=self.activation)
        for lin in self.linears:
            linear_init(lin, activation=self.activation)
        linear_init(self.out)


# class SelfAttention(nn.Module):
#     def __init__(self, input_dim, k_dim,v_dim, head = 1, dropout = 0):
#         super().__init__()
#         self.k_dim = k_dim
#         self.v_dim = v_dim
#         self.head = head
#         self.linear_q = nn.Linear(input_dim, k_dim, bias=False)
#         self.linear_k = nn.Linear(input_dim, k_dim, bias=False)
#         self.linear_v = nn.Linear(input_dim, v_dim, bias=False)
#         self.output_linear = nn.Linear(v_dim, v_dim)
#         self.dropout = nn.Dropout(p = dropout) if dropout > 0 else nn.Identity()
#         self.init_weight()
#     def init_weight(self):
#         for m in self.modules():
#             if isinstance(m,nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, a=1)
#                 # nn.init.constant_(m.bias, 0)
#     def forward(self,x):
#         # x dim: B N C
#         _,N,C = x.shape
#         q = self.linear_q(x)
#         k = self.linear_k(x)
#         v = self.linear_v(x)
#         if self.head != 1:
#             q = q.view(1, N, self.head, self.k_dim // self.head).transpose(1, 2)
#             k = k.view(1, N, self.head, self.k_dim // self.head).transpose(1, 2)
#             v = v.view(1, N, self.head, self.v_dim // self.head).transpose(1, 2)
#         affinity_matrix = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.k_dim // self.head)
#         attention = torch.softmax(affinity_matrix,dim=-1)
#         attention = self.dropout(attention)
#         context = torch.matmul(attention,v)
#         if self.head != 1:
#             context = context.transpose(1,2).contiguous().view(1,N,-1)
#         attention_out = self.output_linear(context)
#         return attention_out

# class CrosssAttention(nn.Module):
#     def __init__(self, k_input_dim, v_input_dim, k_dim,v_dim, head = 1, dropout = 0):
#         super().__init__()
#         self.k_dim = k_dim
#         self.v_dim = v_dim
#         self.head = head
#         self.linear_q = nn.Linear(k_input_dim, k_dim, bias=False)
#         self.linear_k = nn.Linear(k_input_dim, k_dim, bias=False)
#         self.linear_v = nn.Linear(v_input_dim, v_dim, bias=False)
#         self.output_linear = nn.Linear(v_dim, v_dim)
#         self.dropout = nn.Dropout(p = dropout) if dropout > 0 else nn.Identity()
#         self.init_weight()
#
#     def init_weight(self):
#         for m in self.modules():
#             if isinstance(m,nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, a=1)
#                 # nn.init.constant_(m.bias, 0)
#
#     def forward(self,q_x, k_x, v_x):
#         # x dim: B N C
#         _,Bq,Cq = q_x.shape
#         _,Bs,Cv = v_x.shape
#         q = self.linear_q(q_x)
#         k = self.linear_k(k_x)
#         # v = self.linear_v(v_x)
#         # B * N * 8 * 512 / 8
#         v = v_x
#         if self.head != 1:
#             q = q.view(1, Bq, self.head, self.k_dim // self.head).transpose(1,2)
#             k = k.view(1, Bs, self.head, self.k_dim // self.head).transpose(1,2)
#             v = v.view(1, Bs, self.head, self.v_dim // self.head).transpose(1,2)
#             # print(v_x.shape)
#         # print(q.shape)
#         # print(k.shape)
#         # print(v.shape)
#
#         affinity_matrix = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.k_dim // self.head)
#         attention = torch.softmax(affinity_matrix,dim=-1)
#         attention = self.dropout(attention)
#         context = torch.matmul(attention,v)
#         if self.head != 1:
#             context = context.transpose(1,2).contiguous().view(1,Bq,-1)
#         attention_out = self.output_linear(context)
#         return context

# class MLP(nn.Module):
#     def __init__(self,r_dim, z_dim, q_dim, class_num, temper = 0.1):
#         super().__init__()
#         # self.proj_rz = nn.Linear(r_dim + z_dim, q_dim, bias = False)
#         all_dim = q_dim + r_dim
#         self.class_num = class_num
#         self.linear1 = nn.Linear(in_features=all_dim, out_features=512,bias = False)
#         self.linear2 = nn.Linear(in_features=512, out_features = 64,bias = False)
#         self.output_layer = nn.Linear(in_features=64,out_features=self.class_num,bias = False)
#         self.temper = temper
#         self.init_weight()
#     def init_weight(self):
#         for m in self.modules():
#             if isinstance(m,nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, a=1)
#                 # nn.init.constant_(m.bias, 0)
#
#     def forward(self, query, r, z = None):
#         # query :1 Bq 1024 r:1 Bq 256
#         if z:
#             expand_z = z.expand(-1, r.shape[1], -1)
#             rz_merge = torch.cat([expand_z, r],dim = -1)
#             rz_merge = self.proj_rz(rz_merge)
#         else:
#             rz_merge = r
#         # combine_feats:1 Bq 1280
#         combine_feats = torch.cat([rz_merge, query],dim = -1)
#         linear1_feats = torch.sigmoid(self.linear1(combine_feats))
#         linear2_feats = torch.sigmoid(self.linear2(linear1_feats))
#         prob = self.output_layer(linear2_feats).squeeze(0)
#         # 添加温度系数
#         prob = torch.softmax(prob / self.temper,dim=-1)
#         return prob



