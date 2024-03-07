import torch

from lightning.utils.configs import configurable
from .base import BaseBalancer
from typing import List
from lightning.balancer.regist_balancer import BALANCER_REGISTRY


@BALANCER_REGISTRY.register()
class FixWeight(BaseBalancer):

    @configurable
    def __init__(
        self, 
        *,
        init_weights,
        is_sum_loss=True,
        **kwargs):
        super(FixWeight, self).__init__(**kwargs)
        self.init_weight = init_weights
        self.is_sum_loss = is_sum_loss
        assert len(self.init_weight) == self.num_task, \
            "cfg.balancer.weight length {}, but has {} tasks.".format(len(self.init_weight), self.num_task)
        # self.params = torch.tensor(self.init_weight, device=self.device)
        self.register_buffer("params", torch.tensor(self.init_weight))
    
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        if isinstance(cfg.balancer.init_weight, float):
            ret["init_weights"] = cfg.balancer.init_weight
        else:
            ret["init_weights"] = [cfg.balancer.init_weight[t] for t in ret['task_names']]
        if "is_sum_loss" in cfg.balancer:
            ret["is_sum_loss"] = cfg.balancer.is_sum_loss
        return ret

    def run(self, losses, **kwargs):
        losses = super(FixWeight, self).run(losses, **kwargs)
        if self.is_sum_loss:
            return torch.sum(losses * self.params)
        return {
            "loss": losses * self.params
        }


@BALANCER_REGISTRY.register()
class EqualWeight(FixWeight):
    
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["init_weights"] = [ret["init_weights"]] * ret["num_tasks"]
        return ret


class SumOneWeight(FixWeight):

    def coding_weights(self) -> List:
        w = self.cfg.balancer.weight
        assert self.num_task == 2, "SumOneWeight Only Support 2 tasks."
        assert 0 <= w <= 1, "SumOneWeight Only Support weight in [0, 1]"
        return [w, 1.0-w]






