# -*- coding: utf-8 -*-
# @Time : 2021/4/1 下午2:27
# @Author : lyz
# @Email : sqlyz@hit.edu.cn
# @File : base
# @Project : multitask
import torch
from torch import nn


class BaseBackbone(nn.Module):

    def __init__(self, cfg):
        super(BaseBackbone, self).__init__()
        self.cfg = cfg
        self.register_buffer("pixel_mean", torch.Tensor(cfg.train_setting.pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.train_setting.pixel_std).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, data, *args, **kwargs):
        data_to_device = data.to(self.device)


