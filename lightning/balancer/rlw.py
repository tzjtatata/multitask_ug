# This method adopted from paper: arXiv:2111.10603v1
from random import randint, random
import torch
from torch import nn
from .base import BaseBalancer
from lightning.utils.configs import configurable
from lightning.balancer.regist_balancer import BALANCER_REGISTRY


def generate_bernoulli(num_task):
    rlweights = torch.bernoulli(torch.ones(num_task)*0.5)
    while torch.sum(rlweights) == 0:
        rlweights = torch.bernoulli(torch.ones(num_task)*0.5)
    rlweights = rlweights / torch.sum(rlweights)
    return rlweights


def generate_constraint_bernoulli(num_task):
    rlweights = torch.zeros(num_task)
    rlweights[randint(0, num_task-1)] = 1
    # rlweights = rlweights / torch.sum(rlweights)
    return rlweights


def generate_uniform(num_task):
    rlweights = torch.ones(num_task).uniform_()
    return torch.softmax(rlweights, dim=0)


def generate_normal(num_task):
    rlweights = torch.ones(num_task).normal_()
    return torch.softmax(rlweights, dim=0)


def generate_dirichlet(num_task):
    # Need torch > 1.10
    rlweights = torch.distributions.Dirichlet(torch.ones(num_task)).sample()
    return torch.softmax(rlweights, dim=0)


def generate_random_normal(num_task, mean, std):
    rlweights = torch.normal(mean=torch.tensor(mean), std=torch.tensor(std))
    return torch.softmax(rlweights, dim=0)


DIST_MAP = {
    "Bernoulli": generate_bernoulli, 
    "ConstraintBernoulli": generate_constraint_bernoulli,
    "Uniform": generate_uniform,
    "Normal": generate_normal,
    "Dirichlet": generate_dirichlet,
    "RandomNormal": generate_random_normal
}


@BALANCER_REGISTRY.register()
class RLW(BaseBalancer):

    @configurable
    def __init__(self, 
        dist_type,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dist_type = dist_type
        self.register_buffer('params', torch.ones(self.num_task))
        self.means = None
        self.stds = None
        if self.dist_type == 'RandomNormal':
            self.means = [random() for _ in range(self.num_task)]
            self.stds = [random() for _ in range(self.num_task)]
    
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['dist_type'] = cfg.balancer.dist_type
        return ret
    
    def sample_weights(self):
        infos = {}
        if self.dist_type == 'RandomNormal': infos['mean'], infos['std'] = self.means, self.stds
        self.params.data = DIST_MAP[self.dist_type](self.num_task, **infos).to(self.params.device)
    
    def run(self, losses, **kwargs):
        order_losses = super().run(losses, **kwargs)  # should be Tensor (num_task, )
        self.sample_weights()
        loss = order_losses * self.params
        return torch.sum(loss)
