# -*- coding: utf-8 -*-
import os
import torch
from .base import BaseBalancer
from torch import nn
from lightning.utils.configs import configurable
from lightning.balancer.regist_balancer import BALANCER_REGISTRY


@BALANCER_REGISTRY.register()
class ImpMTL(BaseBalancer):

    @configurable
    def __init__(
            self,
            a,
            b,
            do_loss_balance, 
            do_grad_balance, 
            **kwargs
    ):
        super(ImpMTL, self).__init__(**kwargs)
        self.params = nn.Parameter(torch.ones(self.num_task), requires_grad=True)
        self.need_grad = False
        self._optim = None
        self.need_optimizer = True
        self.a = a
        self.b = b
        self.do_loss_balance = do_loss_balance
        self.do_grad_balance = do_grad_balance
        self.need_outer_optimize = True
        self._grad_weights = None
        self.is_imtl = True

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['a'] = cfg.balancer.a
        ret['b'] = cfg.balancer.b
        ret['do_loss_balance'] = cfg.balancer.do_loss_balance
        ret["do_grad_balance"] = cfg.balancer.do_grad_balance
        return ret

    def set_optim(self, optim):
        assert self._optim is None
        self._optim = optim

    def run(self, losses, **kwargs):
        # loss balance
        loss = torch.stack(losses)
        if self.do_loss_balance:
            loss = self.b * (self.a ** self.params) * loss - self.params
        return loss
    
    def get_weights(self):
        ret = {}
        if self.do_loss_balance:
            _w = self.b * (self.a ** self.params.detach())
            for i, t in enumerate(self.task_name):
                ret["{}_weights".format(t)] = _w[i].item()
        if self.do_grad_balance and self._grad_weights is not None:
            for i, t in enumerate(self.task_name):
                ret["{}_grad_weights".format(t)] = self._grad_weights[i].item()
        return ret

    def grad_balance(self, losses, z):
        # 计算grad以及grad范数
        grads = []
        g_norms = []
        for i in range(losses.shape[0]):
            g = torch.autograd.grad(losses[i], z, retain_graph=True)[0].flatten()
            # grad = torch.flatten(torch.autograd.grad(loss, z, retain_graph=True, )[0])
            g_norm = torch.norm(g)
            grads.append(g)
            g_norms.append(g / g_norm)
        # 计算D和U进行gradient balance
        g0, u0 = grads[0], g_norms[0]
        D = g0 - torch.stack(grads[1:])
        U = u0 - torch.stack(g_norms[1:])
        # 求参数alpha
        alpha_sta = torch.matmul(g0, U.T)
        try:
            alpha_end = torch.linalg.inv(torch.matmul(D, U.T))
        except:
            alpha_end = torch.linalg.inv(torch.matmul(D, U.T))
        alpha_ = torch.matmul(alpha_sta, alpha_end)
        _alpha_ = (1 - alpha_.sum()).unsqueeze(0)
        # print(alpha_, _alpha_)
        alpha = torch.cat((_alpha_, alpha_))

        self._grad_weights = alpha.detach().clone()
        # del D, U
        return alpha.detach()

    def _set_grad(self, losses, shared):
        if self.do_grad_balance:
            alpha = self.grad_balance(losses, shared)
            # 改变task_specific parameter和loss scale parameter
            shared_params, task_params, balancer_params = None, None, None
            for pg in self._optim.param_groups:
                if pg['type'] == 'backbone':
                    shared_params = pg['params']
                if pg['type'] == 'head':
                    task_params = pg['params']
                if pg['type'] == 'balancer':
                    balancer_params = pg['params']
            shared_loss = torch.sum(alpha * losses)
            shared_loss.backward(retain_graph=True, inputs=shared_params,)
            loss = torch.sum(losses)
            loss.backward(retain_graph=True, inputs=task_params)
            loss.backward(retain_graph=True, inputs=balancer_params)
        else:
            loss = torch.sum(losses)
            loss.backward()
        return


@BALANCER_REGISTRY.register()
class AuxImpMTL(ImpMTL):

    @configurable
    def __init__(
            self,
            main_tasks,
            aux_tasks,
            **kwargs
    ):
        super(AuxImpMTL, self).__init__(**kwargs)
        self.aux_tasks = aux_tasks
        self.main_tasks = main_tasks
        self.aux_task_num = len(self.aux_tasks)
        self.main_task_num = len(self.main_tasks)
        self.lambda_w = nn.Parameter(torch.log(torch.ones(self.aux_task_num) * (1.0 / (self.main_task_num+self.aux_task_num))), requires_grad=True)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['main_tasks'] = cfg.dataset.task.tasks
        ret['aux_tasks'] = cfg.dataset.task.aux_tasks
        return ret

    def grad_balance(self, losses, z):
        # 计算grad以及grad范数
        main_g = []
        main_u = []
        aux_g = []
        for i in range(self.main_task_num):
            g = torch.autograd.grad(losses[i], z, retain_graph=True)[0].flatten()
            # grad = torch.flatten(torch.autograd.grad(loss, z, retain_graph=True, )[0])
            g_norm = torch.norm(g)
            main_g.append(g)
            main_u.append(g / g_norm)
        for i in range(self.aux_task_num):
            g = torch.autograd.grad(losses[self.main_task_num+i], z, retain_graph=True)[0].flatten()
            aux_g.append(g)
        # 计算D和U进行gradient balance
        g1, u1 = main_g[0], main_u[0]
        D = g1 - torch.stack(main_g[1:])
        U = u1 - torch.stack(main_u[1:])
        sum_of_aux_g = torch.sum(torch.stack(aux_g) * torch.exp(self.lambda_w), dim=0)
        # 求参数alpha
        alpha_sta = torch.matmul(g1 - sum_of_aux_g, U.T)  # in v1 and v2,这里的符号错了，V1和V2里这个符号是+，而其实按照道理应该是-
        try:
            alpha_end = torch.linalg.inv(torch.matmul(D, U.T))
        except:
            alpha_end = torch.linalg.inv(torch.matmul(D, U.T))
        alpha_ = torch.matmul(alpha_sta, alpha_end)
        _alpha_ = (1 - alpha_.sum()).unsqueeze(0)
        # print(alpha_, _alpha_)
        # alpha = torch.cat((_alpha_, alpha_, self.lambda_w))  # 3-24 version which reaches -2.21%.
        alpha = torch.cat((_alpha_, alpha_, torch.exp(self.lambda_w)))
        self._grad_weights = alpha.detach().clone()
        # del D, U
        # 这里包含了需要更新的参数：self.lambda_w
        return alpha
    
    def get_weights(self):
        ret = {}
        if self.do_loss_balance:
            _w = self.b * (self.a ** self.params.detach())
            for i, t in enumerate(self.main_tasks):
                ret["{}_weights".format(t)] = _w[i].item()
        if self.do_grad_balance and self._grad_weights is not None:
            for i, t in enumerate(self.main_tasks):
                ret["{}_grad_weights".format(t)] = self._grad_weights[i].item()
            for i, t in enumerate(self.aux_tasks):
                ret["{}_aux_weights".format(t)] = torch.exp(self.lambda_w[i]).item()
        return ret
    
    def _set_grad(self, losses, shared):
        if self.do_grad_balance:
            alpha = self.grad_balance(losses, shared)
            # 改变task_specific parameter和loss scale parameter
            shared_params, task_params, balancer_params = None, None, None
            for pg in self._optim.param_groups:
                if pg['type'] == 'backbone':
                    shared_params = pg['params']
                if pg['type'] == 'head':
                    task_params = pg['params']
            shared_loss = torch.sum(alpha * losses)
            shared_loss.backward(retain_graph=True, inputs=shared_params,)
            shared_loss.backward(retain_graph=True, inputs=self.lambda_w,)
            loss = torch.sum(losses)
            loss.backward(retain_graph=True, inputs=task_params)
            loss.backward(retain_graph=True, inputs=self.params)
        else:
            loss = torch.sum(losses)
            loss.backward()
        return


@BALANCER_REGISTRY.register()
class ImpMTL2(BaseBalancer):

    @configurable
    def __init__(
            self,
            a,
            b,
            do_loss_balance,
            do_grad_balance,
            use_parameters,
            **kwargs
    ):
        super(ImpMTL2, self).__init__(**kwargs)
        self.params = nn.Parameter(torch.ones(self.num_task), requires_grad=True)
        if use_parameters:
            self.use_last_layer = True
        self.need_grad = False
        self._optim = None
        self.need_optimizer = True
        self.a = a
        self.b = b
        self.do_loss_balance = do_loss_balance
        self.do_grad_balance = do_grad_balance
        self.need_outer_optimize = True
        self._grad_weights = None
        self.is_imtl = True

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['a'] = cfg.balancer.a
        ret['b'] = cfg.balancer.b
        ret['do_loss_balance'] = cfg.balancer.do_loss_balance
        ret["do_grad_balance"] = cfg.balancer.do_grad_balance
        ret['use_parameters'] = cfg.balancer.use_parameters
        return ret

    def set_optim(self, optim):
        assert self._optim is None
        self._optim = optim

    def run(self, losses, **kwargs):
        # loss balance
        loss = torch.stack(losses)
        if self.do_loss_balance:
            loss = self.b * (self.a ** self.params) * loss - self.params
        return loss

    def get_weights(self):
        ret = {}
        if self.do_loss_balance:
            _w = self.b * (self.a ** self.params.detach())
            for i, t in enumerate(self.task_name):
                ret["{}_weights".format(t)] = _w[i].item()
        if self.do_grad_balance and self._grad_weights is not None:
            for i, t in enumerate(self.task_name):
                ret["{}_grad_weights".format(t)] = self._grad_weights[i].item()
        return ret

    def grad_balance(self, losses, z):
        # 计算grad以及grad范数
        grads = []
        g_norms = []
        for i in range(losses.shape[0]):
            g = torch.autograd.grad(losses[i], z, retain_graph=True)[0]
            g_size = g.shape
            g = g.flatten()
            # grad = torch.flatten(torch.autograd.grad(loss, z, retain_graph=True, )[0])
            g_norm = torch.norm(g)
            grads.append(g)
            g_norms.append(g / g_norm)
        # 计算D和U进行gradient balance
        g0, u0 = grads[0], g_norms[0]
        D = g0 - torch.stack(grads[1:])
        U = u0 - torch.stack(g_norms[1:])
        # 求参数alpha
        alpha_sta = torch.matmul(g0, U.T)
        try:
            alpha_end = torch.linalg.inv(torch.matmul(D, U.T))
        except:
            alpha_end = torch.linalg.inv(torch.matmul(D, U.T))
        alpha_ = torch.matmul(alpha_sta, alpha_end)
        _alpha_ = (1 - alpha_.sum()).unsqueeze(0)
        # print(alpha_, _alpha_)
        alpha = torch.cat((_alpha_, alpha_))
        g0 = g0 * _alpha_.item()
        for i in range(1, len(grads)):
            g0 += alpha[i].item() * grads[i]
        self._grad_weights = alpha.detach().clone()
        # del D, U
        return alpha.detach(), g0.reshape(g_size)

    def _set_grad(self, losses, shared):
        if self.do_grad_balance:
            alpha, imtl_grad = self.grad_balance(losses, shared)
            # 改变task_specific parameter和loss scale parameter
            shared_params, task_params, balancer_params = None, None, None
            for pg in self._optim.param_groups:
                if pg['type'] == 'backbone':
                    shared_params = pg['params']
                if pg['type'] == 'head':
                    task_params = pg['params']
                if pg['type'] == 'balancer':
                    balancer_params = pg['params']
            shared.backward(imtl_grad, retain_graph=True, inputs=shared_params)
            loss = torch.sum(losses)
            loss.backward(retain_graph=True, inputs=task_params)
            loss.backward(retain_graph=True, inputs=balancer_params)
            del imtl_grad
        else:
            loss = torch.sum(losses)
            loss.backward()
        return


@BALANCER_REGISTRY.register()
class UNImpMTL2(ImpMTL2):

    @configurable
    def __init__(
            self,
            main_tasks,
            aux_tasks,
            is_scale_aux_w,
            map_reverse,
            map_smooth,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.aux_tasks = aux_tasks
        self.main_tasks = main_tasks
        self.aux_task_num = len(self.aux_tasks)
        self.main_task_num = len(self.main_tasks)
        self.lambda_w = nn.Parameter(-0.5 * torch.log(2.0 * torch.ones(self.main_task_num + self.aux_task_num)),
                                     requires_grad=True)
        self.register_buffer("aux_weights", torch.ones(self.aux_task_num))
        self.is_scale_aux_w = is_scale_aux_w
        self.map_reverse = map_reverse
        self.map_smooth = map_smooth
        self.params.requires_grad = False
        self._cur_ratio = None

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['main_tasks'] = cfg.dataset.task.tasks
        ret['aux_tasks'] = cfg.dataset.task.aux_tasks
        ret['is_scale_aux_w'] = cfg.balancer.is_scale_aux_w
        ret['map_reverse'] = cfg.balancer.map_reverse
        ret['map_smooth'] = cfg.balancer.map_smooth
        return ret

    @property
    def weights(self):
        # wi = 0.5 * exp(-s)
        return 0.5 * torch.exp(-2.0 * self.lambda_w)

    @property
    def regular_term(self):
        return self.lambda_w

    @property
    def theta(self):
        return torch.exp(self.lambda_w[self.main_task_num:]).detach()

    def run(self, losses, **kwargs):
        # loss balance
        loss = torch.stack(losses)
        if self.do_loss_balance:
            loss = self.weights * loss
        return loss

    def grad_balance(self, losses, z):
        # 计算grad以及grad范数
        main_g = []
        main_u = []
        aux_g = []
        g_size = z.shape
        for i in range(self.main_task_num):
            g = torch.autograd.grad(losses[i], z, retain_graph=True)[0].flatten()
            # grad = torch.flatten(torch.autograd.grad(loss, z, retain_graph=True, )[0])
            g_norm = torch.norm(g)
            main_g.append(g)
            main_u.append(g / g_norm)
        for i in range(self.aux_task_num):
            g = torch.autograd.grad(losses[self.main_task_num + i], z, retain_graph=True)[0].flatten()
            aux_g.append(g)
        # 计算D和U进行gradient balance
        if len(main_g) > 1:
            g1, u1 = main_g[0], main_u[0]
            D = g1 - torch.stack(main_g[1:])
            U = u1 - torch.stack(main_u[1:])
            # print("Current Lambda_w is ", self.lambda_w)
            # sum_of_aux_g = torch.sum(torch.stack(aux_g) * self.lambda_w, dim=0)
            # 求参数alpha
            alpha_sta = torch.matmul(g1, U.T)
            try:
                alpha_end = torch.linalg.inv(torch.matmul(D, U.T))
            except:
                alpha_end = torch.linalg.inv(torch.matmul(D, U.T))
            alpha_ = torch.matmul(alpha_sta, alpha_end)
            _alpha_ = (1 - alpha_.sum()).unsqueeze(0)
            main_norm = torch.norm(_alpha_.squeeze().item() * main_g[0])
            alpha = torch.cat((_alpha_, alpha_))
            imtl_grad = g1 * _alpha_.item()
            for i in range(1, len(main_g)):
                imtl_grad += alpha[i].item() * main_g[i]
        else:
            main_norm = torch.norm(main_g[0])
            alpha = torch.ones(len(main_g), device=main_g[0].device)
            imtl_grad = main_g[0]

        # update weights via theta.
        aux_w = 1.0 - self.theta
        if self.is_scale_aux_w and self._cur_ratio is not None:
            self._cur_ratio = (1.0 - self.map_smooth) * self._cur_ratio + self.map_smooth * aux_w
        else:
            self._cur_ratio = aux_w

        for i in range(len(aux_g)):
            lambda_w = main_norm.item() / torch.norm(aux_g[i]).item()
            self.aux_weights[i] = lambda_w * self._cur_ratio[i].item()

            g_product = imtl_grad * aux_g[i]
            aux_g[i][g_product < 0] = 0.0

        for i in range(len(aux_g)):
            imtl_grad += aux_g[i] * self.aux_weights.data[i].item()
        alpha = torch.cat((alpha, self.aux_weights))
        self._grad_weights = alpha.detach().clone()
        # del D, U
        # 这里包含了需要更新的参数：self.lambda_w
        # return alpha # v1中允许主任务往lambda_w传参
        return alpha.detach(), imtl_grad.reshape(g_size)  # v2不允许主任务传参。

    def get_weights(self):
        ret = {}
        if self.do_loss_balance:
            _w = self.weights
            for i, t in enumerate(self.main_tasks):
                ret["{}_weights".format(t)] = _w[i].item()
        if self.do_grad_balance and self._grad_weights is not None:
            for i, t in enumerate(self.main_tasks):
                ret["{}_grad_weights".format(t)] = self._grad_weights[i].item()
            for i, t in enumerate(self.aux_tasks):
                ret["{}_aux_weights".format(t)] = self.aux_weights[i].item()
            for i, t in enumerate(self.aux_tasks):
                ret["{}_theta".format(t)] = self._cur_ratio[i].item()
        return ret

    def _set_grad(self, losses, shared):
        if self.do_grad_balance:
            alpha, imtl_grad = self.grad_balance(losses, shared)
            shared_params, task_params, balancer_params = None, None, None
            for pg in self._optim.param_groups:
                if pg['type'] == 'backbone':
                    shared_params = pg['params']
                if pg['type'] == 'head':
                    task_params = pg['params']
                if pg['type'] == 'balancer':
                    balancer_params = pg['params']
            shared.backward(imtl_grad, retain_graph=True, inputs=shared_params)
            loss = torch.sum(losses) + torch.sum(self.regular_term)
            loss.backward(retain_graph=True, inputs=task_params)
            loss.backward(retain_graph=True, inputs=self.lambda_w)
        else:
            loss = torch.sum(losses) + torch.sum(self.regular_term)
            loss.backward()
        return

