# -*- encoding: utf-8 -*-
"""
File model_weights.py
Created on 2023/7/12 18:25
Copyright (c) 2023/7/12
@author: 
"""
import torch
import os
from loguru import logger

def save_weights(network, save_path, optimizer = None, scheduler = None, epoch = None, save_best = False):
    os.makedirs(save_path, exist_ok = True)
    if save_best:
        weights_path = os.path.join(save_path, f"weights_best.pth")
        weights = network.module.state_dict()
    else:
        weights_path = os.path.join(save_path, f"weights_{epoch}.pth")
        weights = {
            "epoch" : epoch,
            "network" : network.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler" : scheduler.state_dict()
        }
    torch.save(weights, weights_path)
    logger.info(f"Weight saved to {weights_path}")

def load_weights(network, weights_path, local_rank,optimzer=None, scheduler=None, just_weight=False):
    map_location = "cuda:%d" % local_rank
    weights = torch.load(weights_path,map_location={'cuda:0': map_location})
    if just_weight:
        network.module.load_state_dict(weights)
        return network
    else:
        epoch = weights['epoch']
        net_state_dict = weights['network']
        opt_state_dict = weights['optimzer']
        sch_state_dict = weights['scheduler']
        network.module.load_state_dict(net_state_dict)
        optimzer.load_state_dict(opt_state_dict)
        scheduler.load_state_dict(sch_state_dict)
        logger.info("Network weights, optimizer states, and scheduler states loaded.")
        return network,optimzer,scheduler,epoch
