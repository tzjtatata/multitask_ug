# -*- coding: utf-8 -*-
# @Time : 10/16/21 11:58 PM
# @Author : lyz
# @Email : sqlyz@hit.edu.cn
# @File : build_optim
# @Project : multitask
from torch.optim import SGD, Adam


OPTIM_MAP = {
    'SGD': SGD,
    'ADAM': Adam
}


def build_optimizer_by_type(cfg, param_groups, optim_type, **kwargs):
    if optim_type == 'SGD':
        optimizer = SGD(
            param_groups,
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"]
        )
    elif optim_type == 'ADAM':
        optimizer = Adam(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=cfg["weight_decay"]
        )
    else:
        raise NotImplementedError("Not Support Other Optimizer.")
    return optimizer


def build_optimizer(param_groups, optimizer_type, **kwargs):
    return OPTIM_MAP[optimizer_type](
        param_groups,
        **kwargs
    )
