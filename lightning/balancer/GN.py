import torch
from .base import BaseBalancer
from torch import nn
from lightning.utils.configs import configurable
from lightning.utils.build_optim import build_optimizer_by_type
from lightning.balancer.regist_balancer import BALANCER_REGISTRY


@BALANCER_REGISTRY.register()
class GradNorm(BaseBalancer):

    @configurable
    def __init__(
        self, 
        alpha,
        balancer_type,
        lr,
        optimizer_cfg,
        **kwargs
    ):
        super(GradNorm, self).__init__(**kwargs)
        self.params = nn.Parameter(torch.ones(self.num_task, device=self.device), requires_grad=False)
        self.need_grad = True
        self.alpha = alpha
        self.init_loss = None
        self.current_loss = None
        self.optimizer = build_optimizer_by_type(
            optimizer_cfg,
            [{'params': self.params, 'type': 'balancer', 'lr': lr}],
            balancer_type
        )
    
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['alpha'] = cfg.balancer.alpha
        ret['lr'] = cfg.balancer.lr
        ret['balancer_type'] = cfg.balancer.optimizer
        ret['optimizer_cfg'] = dict(cfg.dataset.solver.optimizer)
        return ret

    def run(self, losses, grads=None, **kwargs):
        loss = super(GradNorm, self).run(losses)
        self.current_loss = loss.data.clone()

        assert grads is not None
        if hasattr(self, 'scales'):
            grads = [
                grads[i] * self.scales[i]
                for i in range(len(grads))
            ]
        self.cur_grads = grads

        loss = self.weights * loss
        return torch.sum(loss)

    def before_bp(self):
        self.optimizer.zero_grad()

    def cal_loss(self):
        grads = self.cur_grads
        g_norms = [torch.norm(self.weights[i] * grads[i])
                   for i in range(len(self.task_name))]

        # debug
        # g_debug = [g.item() for g in g_norms]
        # print("Grad Norm: ", g_debug)

        g_norms = torch.stack(g_norms)

        loss_ratio = self.current_loss / self.init_loss
        # print("loss_ratio:", loss_ratio)
        inverse_train_rate = loss_ratio / torch.mean(loss_ratio)
        # print("inverse_training_rate:", inverse_train_rate)

        # At the end of chapter 3.2, the paper declaim that gradient will only backward from g_norms, not from mean g_norms.
        constant_term = torch.mean(g_norms).detach() * (inverse_train_rate ** self.alpha)

        l_grad = torch.sum(torch.abs(g_norms - constant_term))
        # print("L_grad:", l_grad.item())
        return l_grad

    def after_optim(self):
        if self.init_loss is None:
            self.init_loss = self.current_loss
            return
        # l_grad = self.cal_loss()
        # self.current_loss = None
        # return
        self.params.requires_grad = True
        self.optimizer.zero_grad()
        l_grad = self.cal_loss()
        # real_weights_grad = torch.autograd.grad(l_grad, self.weights)[0]
        l_grad.backward()

        self.current_loss = None

        self.optimizer.step()
        self.params.requires_grad = False
        self.params.data = torch.relu(self.params) + 0.01

        # normalize weights is the key of GradNorm, if Not Normalize, GradNorm will out of control.
        self.params.data = self.params * len(self.task_name) / torch.sum(self.params)


