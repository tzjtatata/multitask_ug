from .base import BaseBalancer
from lightning.utils.configs import configurable
import torch
from torch import nn


class DWA(BaseBalancer):

    @configurable
    def __init__(
        self,
        T,
        is_sum_loss=True,
        **kwargs
    ):
        super(DWA, self).__init__(**kwargs)
        self.is_sum_loss = is_sum_loss
        self.T = T
        self.params = nn.Parameter(torch.ones(self.num_task, device=self.device), requires_grad=False)
        self.records = []

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['T'] = cfg.balancer.T
        if 'is_sum_loss' in cfg.balancer:
            ret['is_sum_loss'] = cfg.balancer.is_sum_loss
        return ret

    @property
    def weights(self):
        return self.params
    
    def cal_new_weights(self, cur_loss):
        if len(self.records) < 2:
            self.records.append(cur_loss.detach())
            return 
        r_n = self.records[0] / self.records[1]
        exp_r_n = torch.exp(r_n / self.T)
        self.params.data = len(self.task_name) *  exp_r_n / torch.sum(exp_r_n)

        self.records.append(cur_loss.detach())
        self.records = self.records[1:]
        return 

    def run(self, losses, **kwargs):
        loss = super(DWA, self).run(losses)
        self.cal_new_weights(loss)
        loss = self.weights * loss
        if not self.is_sum_loss:
            return {
                "loss": loss
            }
        return torch.sum(loss)