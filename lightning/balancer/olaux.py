# -*- coding: utf-8 -*-
import torch
from .base import BaseAuxBalancer
from torch import nn
from lightning.utils.configs import configurable
from torch.optim import SGD
from lightning.balancer.regist_balancer import BALANCER_REGISTRY


@BALANCER_REGISTRY.register()
class OLAUX(BaseAuxBalancer):
    """
        Only for one main task.
    """

    @configurable
    def __init__(
        self,
        balancer_lr,
        N,
        constrained_weights, 
        **kwargs
    ):
        super(OLAUX, self).__init__(**kwargs)
        self.params = nn.Parameter(torch.ones(len(self.aux_tasks)), requires_grad=True)
        self.need_grad = False
        self.need_optimizer = True
        self.grad_cache = None
        self.constrained_weights = constrained_weights
        self.N = N
        self.N_counter = N
        self._optimizer = SGD([self.params], lr=balancer_lr)
        self._optim = None

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['balancer_lr'] = cfg.balancer.lr
        ret['N'] = cfg.balancer.N
        ret['constrained_weights'] = cfg.balancer.constrained_weights
        return ret
    
    def set_optim(self, optim):
        assert self._optim is None
        self._optim = optim

    def run(self, losses, **kwargs):
        # loss balance
        main_loss = torch.log(losses[0])
        aux_losses = torch.log(torch.stack(losses[1:]))
        weights = torch.relu(self.params) if self.constrained_weights else self.params
        loss = main_loss + torch.sum(weights * aux_losses)
        self.update_weights(main_loss, aux_losses, kwargs['repre'])
        return loss
    
    def get_weights(self):
        ret = {
            self.task_name[0]: 1.0,
        }

        for i, t in enumerate(self.aux_tasks):
            ret["{}_weights".format(t)] = self.params[i].item()
        return ret

    def update_weights(self, main_loss, aux_losses, z):
        # 计算grad以及grad范数
        main_grad  = torch.autograd.grad(main_loss, z, retain_graph=True)[0].reshape(1, -1)

        aux_grads = []
        for i in range(aux_losses.shape[0]):
            g = torch.autograd.grad(aux_losses[i], z, retain_graph=True)[0].flatten()
            aux_grads.append(g)
        aux_grads = torch.stack(aux_grads, dim=0)
        
        # 计算当次梯度，并累加
        alpha = self._optim.param_groups[0]['lr']
        _cur_grad = -alpha * torch.matmul(main_grad, aux_grads.T) 
        if self.grad_cache is None: 
            self.grad_cache =  _cur_grad
        else:
            self.grad_cache += _cur_grad
    
    def after_optim(self):
        if self.N_counter != 0:
            # print("Not Update", self.N_counter)
            self.N_counter -= 1
        else:
            # 更新
            # print("Update~")
            self.N_counter = self.N
            self.params.grad = self.grad_cache.data.flatten()
            self._optimizer.step()
            # print(self.params)
            self._optimizer.zero_grad()
            self.grad_cache = None
