import torch
from torch import nn
from lightning.balancer.base import BaseBalancer
from lightning.utils.configs import configurable
from lightning.balancer.regist_balancer import BALANCER_REGISTRY


@BALANCER_REGISTRY.register()
class DTP(BaseBalancer):

    @configurable
    def __init__(
        self,
        alpha,
        gamma,
        warmup_step,
        *args,
        **kwargs
    ):
        super(DTP, self).__init__(*args, **kwargs)
        self.params = nn.Parameter(
            torch.ones(len(self.task_name), device=self.device),
            requires_grad=False
        )
        self.kpi_history = torch.zeros(len(self.task_name), device=self.device)
        self.alpha = alpha
        self.gamma = gamma
        self.warmup_step = warmup_step
    
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)  # from detectron2.modeling.roi_heads.Res5ROIHeads.from_config(), it use super().from_config(cfg) to call father class from_config
        new_ret = {
            "alpha": cfg.balancer.alpha,
            "gamma": cfg.balancer.gamma,
            "warmup_step": cfg.balancer.warmup_step,
        }
        ret.update(new_ret)
        return ret

    def generate_weights(self, kpi):
        assert kpi is not None
        kpis = torch.stack(kpi).detach()
        kpis = torch.relu(kpis)
        if self.warmup_step > 0:
            self.warmup_step -= 1
            return

        self.kpi_history = self.kpi_history * self.alpha + kpis * (1.0 - self.alpha)
        self.params.data = -1.0 * ((1 - self.kpi_history) ** self.gamma) * torch.log(self.kpi_history)

    def run(self, losses, kpi=None, **kwargs):
        loss = super(DTP, self).run(losses)
        self.generate_weights(kpi)
        return torch.sum(self.params * loss)


if __name__ == "__main__":
    b = DTP(
        task_names=["semseg", "depth"],
        device='cpu',
        alpha=0.1,
        gamma=0.2,
        warmup_step=100
    )
    print(type(b).__name__)