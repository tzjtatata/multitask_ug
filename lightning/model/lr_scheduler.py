# -*- coding: utf-8 -*-
# @Time : 11/2/21 8:24 PM
# @Author : lyz
# @Email : sqlyz@hit.edu.cn
# @File : lr_scheduler
# @Project : multitask
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):

    def __init__(self, optimizer, power=1.0):
        self.power = power
        self.cur_step = 0
        self.max_steps = None
        super(PolynomialLR, self).__init__(optimizer)

    def get_lr(self):
        if self.max_steps is None:
            return self.base_lrs
        return [
            base_lr * pow(1 - (self.cur_step / self.max_steps), self.power)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        values = self.get_lr()
        
        print('')
        print(self.cur_step+1, '/', self.max_steps)
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            print(f"Group {i} LR change to {lr}")

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self.cur_step += 1

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps