# -*- encoding: utf-8 -*-
"""
File base.py
Created on 2023/7/7 21:27
Copyright (c) 2023/7/7
@author: 
"""

import torch
from torch import Tensor
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Dict
import torch.nn as nn
import abc

from model.anp_head import AttentionNeuralProcessHead
from model.canp_head import ConditionalAdaptiveNeuralProcessHead
from model.cnp_head import ConditionalNeuralProcessHead
from model.debug_head import DebugHead
from model.frn_head import FeatureReconstructionHead
from util.tools import weights_init
from model.resnet import resnet50
from model.meta_baseline_head import MetaBaselineHead


class BaseFewShotClassifier(nn.Module):
    """Base class for classifier.

    Args:
        backbone (dict): Config of the backbone.
        neck (dict | None): Config of the neck. Default: None.
        head (dict | None): Config of classification head.
        train_cfg (dict | None): Training config. Default: None.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 backbone,
                 head,
                 class_num=None):
        super().__init__()
        # 加载backbone
        if backbone == "resnet50":
            self.backbone = resnet50()
            # print(self.backbone)
        # 加载预测头
        if head == "meta_baseline_head":
            self.head = MetaBaselineHead()
        elif head == "anp_head":
            self.head = AttentionNeuralProcessHead(x_dim = 2048, x_trans_dim = 512,y_trans_dim = 512,
                                                   cross_out_dim = 512, r_dim = 512, class_num = class_num)
        elif head == "cnp_head":
            self.head = ConditionalNeuralProcessHead(x_dim = 2048, x_trans_dim = 512, y_trans_dim = 512,
                                                        r_dim = 512, class_num = class_num)
        elif head == "debug_head":
            self.head = DebugHead(x_dim=2048,x_trans_dim=512,y_trans_dim=512,class_num=class_num)
        elif head == "frn_head":
            self.head = FeatureReconstructionHead(class_num=class_num)
        elif head == "canp_head":
            self.head = ConditionalAdaptiveNeuralProcessHead(output_size=2048)
        self.meta_test_cfg = None
        # `device_indicator` is used to record runtime device
        # `MetaTestParallel` will use `device_indicator` to
        # recognize the device of model
        self.register_buffer('device_indicator', torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.device_indicator.device

    def get_device(self):
        return self.device_indicator.get_device()

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        return hasattr(self, 'head') and self.head is not None

    def extract_feat(self, img: Tensor) -> Tensor:
        """Directly extract features from the backbone."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, **kwargs):
        """Forward Function."""

    @abstractmethod
    def forward_train(self, **kwargs):
        """Forward training data."""

    @abstractmethod
    def forward_support(self, **kwargs):
        """Forward support data in meta testing."""

    @abstractmethod
    def forward_query(self, **kwargs):
        """Forward query data in meta testing."""

    @staticmethod
    def _parse_losses(losses: Dict) -> Tuple[Dict, Dict]:
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer`): The optimizer of
                runner is passed to `train_step()`. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys:

                - `loss` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - `log_vars` contains all the variables to be sent to the
                  logger.
                - `num_samples` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    @abstractmethod
    def before_meta_test(self, meta_test_cfg: Dict, **kwargs):
        """Used in meta testing.

        This function will be called before the meta testing.
        """

    @abstractmethod
    def before_forward_support(self, **kwargs):
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """

    @abstractmethod
    def before_forward_query(self, **kwargs):
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """

class BaseAttention(abc.ABC, nn.Module):
    """
    Base Attender module.

    Parameters
    ----------
    kq_size : int
        Size of the key and query.

    value_size : int
        Final size of the value.

    out_size : int
        Output dimension. If not different than `kq_size` will do all the computation
        with a size of `x_dim` and add a linear layer at the end to reshape.

    is_normalize : bool, optional
        Whether weights should sum to 1 (using softmax).

    dropout : float, optional
        Dropout rate to apply to the attention.
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

    def forward(self, keys, queries, values, **kwargs):
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
        logits = self.score(keys, queries, **kwargs)

        attn = self.logits_to_attn(logits)

        attn = self.dropout(attn)

        # attn : size=[batch_size, n_queries, n_keys]
        # values : size=[batch_size, n_keys, value_size]
        context = torch.bmm(attn, values)

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