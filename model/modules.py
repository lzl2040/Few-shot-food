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
        need_hidden = True,
        is_output = False,
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
        self.is_output = is_output
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
        if self.is_output:
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

# canp
class DenseResidualBlock(nn.Module):
    """
    Wrapping a number of residual layers for residual block. Will be used as building block in FiLM hyper-networks.
    :param in_size: (int) Number of features for input representation.
    :param out_size: (int) Number of features for output representation.
    """
    def __init__(self, in_size, out_size):
        super(DenseResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_size, out_size)
        self.linear2 = nn.Linear(out_size, out_size)
        self.linear3 = nn.Linear(out_size, out_size)
        self.elu = nn.ELU()

    def forward(self, x):
        """
        Forward pass through residual block. Implements following computation:

                h = f3( f2( f1(x) ) ) + x
                or
                h = f3( f2( f1(x) ) )

                where fi(x) = Elu( Wi^T x + bi )

        :param x: (torch.tensor) Input representation to apply layer to ( dim(x) = (batch, in_size) ).
        :return: (torch.tensor) Return f(x) ( dim(f(x) = (batch, out_size) ).
        """
        identity = x
        out = self.linear1(x)
        out = self.elu(out)
        out = self.linear2(out)
        out = self.elu(out)
        out = self.linear3(out)
        if x.shape[-1] == out.shape[-1]:
            out += identity
        return out


class LinearClassifierAdaptationNetwork(nn.Module):
    """
    Versa-style adaptation network for linear classifier (see https://arxiv.org/abs/1805.09921 for full details).
    :param d_theta: (int) Input / output feature dimensionality for layer.
    """
    def __init__(self, d_theta):
        super(LinearClassifierAdaptationNetwork, self).__init__()
        self.weight_means_processor = self._make_mean_dense_block(d_theta, d_theta)
        self.bias_means_processor = self._make_mean_dense_block(d_theta, 1)

    @staticmethod
    def _make_mean_dense_block(in_size, out_size):
        """
        Simple method for generating different types of blocks. Final code only uses dense residual blocks.
        :param in_size: (int) Input representation dimensionality.
        :param out_size: (int) Output representation dimensionality.
        :return: (nn.Module) Adaptation network parameters for outputting classification parameters.
        """
        return DenseResidualBlock(in_size, out_size)

    def forward(self, representation_dict):
        """
        Forward pass through adaptation network. Returns classification parameters for task.
        :param representation_dict: (dict::torch.tensors) Dictionary containing class-level representations for each
                                    class in the task.
        :return: (dict::torch.tensors) Dictionary containing the weights and biases for the classification of each class
                 in the task. Model can extract parameters and build the classifier accordingly. Supports sampling if
                 ML-PIP objective is desired.
        """
        classifier_param_dict = {}
        class_weight_means = []
        class_bias_means = []

        # Extract and sort the label set for the task
        label_set = list(representation_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        # For each class, extract the representation and pass it through adaptation network to generate classification
        # params for that class. Store parameters in a list,
        for class_num in label_set:
            nu = representation_dict[class_num]
            class_weight_means.append(self.weight_means_processor(nu))
            class_bias_means.append(self.bias_means_processor(nu))

        # Save the parameters as torch tensors (matrix and vector) and add to dictionary
        classifier_param_dict['weight_mean'] = torch.cat(class_weight_means, dim=0)
        classifier_param_dict['bias_mean'] = torch.reshape(torch.cat(class_bias_means, dim=1), [num_classes, ])

        return classifier_param_dict


