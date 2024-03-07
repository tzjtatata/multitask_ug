# -*- coding: utf-8 -*-
# @Time : 2020/10/16 下午3:28
# @Author : lyz
# @Email : sqlyz@hit.edu.cn
# @File : BasicHead
# @Project : multitask
import torch
from torch import nn

from lightning.utils.configs import configurable


class BasicHead(nn.Module):

    @configurable
    def __init__(
        self, 
        task_name,
        reduction='mean'
    ):
        super().__init__()
        self.task_name = task_name
        self.reduction = reduction
        self.map_func = self.build_head()
    
    @classmethod
    def from_config(cls, cfg, task_name):
        reduction = cfg.balancer.reduction if 'reduction' in cfg.balancer else 'mean'
        return {
            "task_name": task_name.lower(),
            "reduction": reduction
        }

    def build_head(self):
        raise NotImplementedError("You Should Implement build_head function before using this head.")

    def forward(self, data):
        return self.map_func(data)

    def losses(self, pred, gt):
        raise NotImplementedError("You Should Implement a loss function.")

    def kpi(self, pred, gt, eps=1e-6):
        raise NotImplementedError("You Should Implement a kpi function, if you use DTP.")
