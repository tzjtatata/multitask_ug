import torch
from .base import BaseBalancer, BaseAuxBalancer
from torch import nn
from lightning.utils.build_optim import build_optimizer_by_type
from lightning.utils.configs import configurable
from lightning.balancer.regist_balancer import BALANCER_REGISTRY


class UncertaintyWeight(BaseBalancer):

    @configurable
    def __init__(
        self,
        is_sum_loss=True,
        **kwargs
    ):
        super(UncertaintyWeight, self).__init__(**kwargs)
        self.need_outer_optimize = True
        self.is_sum_loss = is_sum_loss
        self.params = nn.Parameter(torch.ones(self.num_task, device=self.device), requires_grad=True)
        self.reverse_weights()

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        if 'is_sum_loss' in cfg.balancer:
            ret['is_sum_loss'] = cfg.balancer.is_sum_loss
        return ret

    def reverse_weights(self):
        self.params.data = -1.0 * torch.log(2 * self.params)

    @property
    def weights(self):
        return 0.5 * torch.exp(-1.0 * self.params)

    @property
    def regular_term(self):
        return 0.5 * self.params

    def run(self, losses, **kwargs):
        loss = super(UncertaintyWeight, self).run(losses)
        loss = self.weights * loss
        if not self.is_sum_loss:
            return {
                "loss": loss,
                "regular_term": self.regular_term
            }
        return torch.sum(loss) + torch.sum(self.regular_term)

@BALANCER_REGISTRY.register()
class UWOrigin(UncertaintyWeight):

    @property
    def weights(self):
        # wi = 0.5 * exp(-s)
        return 0.5 * torch.exp(-2.0 * self.params)

    @property
    def regular_term(self):
        return self.params

    def reverse_weights(self):
        self.params.data = -0.5 * torch.log(2.0 * self.params)

    def get_weights(self):
        w_dict = super(UWOrigin, self).get_weights()
        if len(self.task_name) == self.num_task:
            s_dict = {
                "{}_s".format(k.lower()): self.params[i].item()
                for i, k in enumerate(self.task_name)
            }
            w_dict.update(s_dict)
        return w_dict
