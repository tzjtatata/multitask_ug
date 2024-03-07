# -*- coding: utf-8 -*-
"""
  @Time : 2021/1/6 下午12:25
  @Author : lyz
  @Email : sqlyz@hit.edu.cn
  @File : MultiTaskNet.py
  @Project : multitask
"""
import numpy as np
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from lightning.eval.evaluate_utils import calculate_multi_task_performance
from lightning.model.regist_meta_arch import META_ARCH_REGISTRY
from lightning.eval.evaluate_utils import SUPPORT_EVAL_TASKS
from multitask.model.backbones import build_backbone
from multitask.model.heads import build_heads
from lightning.utils.configs import configurable
from lightning.utils.utils import get_output
import torch.nn.functional as F
from lightning.balancer import build_balancer
from torch import Tensor, nn


def adjust_learning_rate(base_lr, optimizer, epoch, max_epochs):
    """ Adjust the learning rate """

    lr = base_lr

    lambd = pow(1 - (epoch / max_epochs), 0.9)
    lr = lr * lambd
    print("Current Learning Rate is {}".format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class GradInspector:

    def __init__(self, optimizer) -> None:
        self._optim = optimizer
        for pg in optimizer.param_groups:
            if pg['type'] == 'backbone':
                self.pg = pg
                break
        self.similarity_collection = []

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for p in self.pg['params']:
            if p.grad is None: continue
            shape.append(p.grad.shape)
            grad.append(p.grad.clone())
            has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _pack_grad(self, objectives):
        '''
        Adapt from PCGrad https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def step(self, losses: list):
        # 获取梯度
        task_grads, shapes, _ = self._pack_grad(losses)
        assert len(task_grads) == 2
        similarity = torch.dot(task_grads[0], task_grads[1]) / (torch.norm(task_grads[0]) * torch.norm(task_grads[1]))
        self.similarity_collection.append(similarity.item())

    def summary(self):
        return np.array(self.similarity_collection)

    def reset(self):
        self.similarity_collection = []


def check_if_task_not_use(mask):
    bs, num_tasks = mask.shape
    for i in range(num_tasks):
        if torch.sum(mask[:, i]) == 0:
            return False
    return True


@META_ARCH_REGISTRY.register()
class VandenNet(pl.LightningModule):

    @configurable
    def __init__(
            self,
            backbone,
            main_tasks,
            aux_tasks,
            balancer,
            heads,
            dataset_name,
            max_epoch,
            optimizer_cfg,
            single_task_results,
            fix_encoder,
            is_log_gn,
            is_fix_head,
            show_weights
    ):
        super().__init__()
        self.backbone = backbone
        self.main_tasks = main_tasks
        self.aux_tasks = aux_tasks

        pre_eval_tasks = self.main_tasks + self.aux_tasks
        self.eval_tasks = []
        for eval_task in pre_eval_tasks:
            if eval_task in SUPPORT_EVAL_TASKS:
                self.eval_tasks.append(eval_task)
        
        self.inference_tasks = main_tasks + aux_tasks
        self.balancer = balancer
        self.cal_kpi = False
        self.cal_grad = False
        self.heads = nn.ModuleDict(heads)
        self.automatic_optimization = False
        self.max_epoch = max_epoch
        self.optimizer_cfg = optimizer_cfg
        self.dataset_name = dataset_name
        self.single_task_results = single_task_results
        self.fix_encoder = fix_encoder
        self.is_log_gn = is_log_gn
        self.is_fix_head = is_fix_head
        self.show_weights = show_weights

    @classmethod
    def from_config(cls, cfg):
        main_tasks = [t.lower() for t in cfg.dataset.task.tasks]
        aux_tasks = [t.lower() for t in cfg.dataset.task.aux_tasks] if 'aux_tasks' in cfg.dataset.task else []
        single_task_results = dict(cfg.dataset.meta_data.single_task_results)
        is_fix_head = True if (('is_fix_head' in cfg.dataset.model) and cfg.dataset.model.is_fix_head) else False
        if is_fix_head and (len(aux_tasks) == 0): raise ValueError('When No Aux Tasks, the dataset.model.is_fix_head should not be True.')
        return {
            "main_tasks": main_tasks,
            "aux_tasks": aux_tasks,
            "backbone": build_backbone(cfg),
            "balancer": build_balancer(cfg.balancer.type, cfg),
            "heads": build_heads(cfg),
            "dataset_name": cfg.dataset.name,
            "max_epoch": cfg.dataset.solver.max_epoch,
            "optimizer_cfg": dict(cfg.dataset.solver.optimizer),
            "single_task_results": single_task_results,
            "fix_encoder": cfg.dataset.solver.fix_encoder,
            "is_log_gn": cfg.debug.is_log_gn,
            "show_weights": cfg.debug.is_log_weights,
            "is_fix_head": is_fix_head
        }

    def _forward_backbone(self, data):
        features = self.backbone(data)
        return features

    def _forward_heads(self, representation, out_size):
        preds = {}
        for task_name in self.inference_tasks:
            head = self.heads[task_name]
            pred = head(representation)
            if task_name != 'imagenet':
                preds[task_name] = F.interpolate(pred, out_size, mode='bilinear')
            else:
                preds[task_name] = pred
        return preds

    def _forward_by_task(self, x, t):
        out_size = x.size()[2:]
        shared = self._forward_backbone(x)
        head = self.heads[t]
        pred = head(shared)
        pred = F.interpolate(pred, out_size, mode='bilinear')
        return shared, pred

    def forward(self, x):
        out_size = x.size()[2:]
        shared = self._forward_backbone(x)
        preds = self._forward_heads(shared, out_size)
        return preds, shared

    def _forward_loss(self, pred, gt):
        losses = []
        bs = pred[self.main_tasks[0]].shape[0]

        for i, t in enumerate(self.inference_tasks):
            if t == 'detection':
                sum_v = None
                for k, v in pred[t].items():
                    if sum_v is None:
                        sum_v = v
                    elif k.startswith('loss'):
                        sum_v += v
                losses.append(sum_v)
                continue

            losses.append(self.heads[t].losses(pred[t], gt[t]))

        return losses

    def _record_grad_norm(self, losses):
        named_layers = {
            k: v
            for k, v in self.backbone.named_modules()
            if isinstance(v, nn.Conv2d)
        }
        for i, t in enumerate(self.main_tasks):
            self.zero_grad(set_to_none=True)
            losses[i].backward(retain_graph=True)
            g = []
            for k, v in named_layers.items():
                # print(k, v.weight.shape)
                _g = v.weight.grad.clone().flatten()
                g.append(_g)
                self.log("GradNorm/{}/{}".format(t, k), torch.norm(_g), sync_dist=True)
                self.log("GradNormRatio/{}/{}".format(t, k), torch.norm(_g) / (losses[i].item()), sync_dist=True)
            total_norm = torch.norm(torch.cat(g).flatten())
            self.log("GradNorm/{}/total".format(t), total_norm, sync_dist=True)
            self.log("GradNormRatio/{}/total".format(t), total_norm / (losses[i].item()), sync_dist=True)

    def training_step(self, batch, batch_idx):
        if self.balancer.need_optimizer and self.balancer._optim is None:
            self.balancer.set_optim(self.optimizers())
        images = batch['image']
        targets = {task: batch[task] for task in self.inference_tasks}

        if self.balancer.is_mgda:
            self.balancer.cal_weights(self, images, targets)

        output, shared = self(images)

        # Version 1.
        # losses = []
        # for t in self.task_names:
        #     losses.append(self.heads[t].losses(output[t], targets[t]))
        #     self.log("{}_loss".format(t),
        #             losses[-1].item(), prog_bar=True, sync_dist=True)

        losses = self._forward_loss(output, targets)
        if self.is_log_gn:
            self._record_grad_norm(losses)
        for i, t in enumerate(self.inference_tasks):
            self.log("{}_loss".format(t),
                    losses[i].item(), prog_bar=True, sync_dist=True)

        addition_info = {
            'repre': shared
        }
        if self.balancer.need_grad:
            layer = self.backbone.get_last_layer()
            grads = [
                torch.autograd.grad(task_loss, layer.parameters(), retain_graph=True)[0].detach()
                for task_loss in losses
            ]
            addition_info['grads'] = grads

        loss = self.balancer.run(losses, **addition_info)

        opt = self.optimizers() if not self.balancer.is_pcgrad else self.balancer
        # Backward
        # opt.zero_grad()
        self.zero_grad()
        opt.zero_grad()
        self.balancer.before_bp()
        self.manual_backward(loss, repre=shared)
        self.balancer.after_bp()
        opt.step()
        self.balancer.after_optim()

        if self.balancer.is_pcgrad:
            loss = torch.sum(torch.stack(loss))
        if self.balancer.is_imtl:
            loss = torch.sum(loss)
        self.log("total_loss", loss.item(), sync_dist=True, prog_bar=True)
        self.log_dict(self.balancer.get_weights(), prog_bar=self.show_weights, sync_dist=True)
        return loss

    def manual_backward(self, loss: Tensor, *args, **kwargs) -> None:
        if self.balancer.is_imtl:
            if self.balancer.use_last_layer:
                self.balancer._set_grad(loss, self.backbone.get_last_layer().weight)
            else:
                self.balancer._set_grad(loss, kwargs['repre'])
            return

        if self.balancer.is_pcgrad:
            self.balancer.pc_backward(loss)
            return
        loss.backward()

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs = batch['image']
        targets = {task: batch[task] for task in self.eval_tasks}
        img_size = (inputs.size(2), inputs.size(3))
        output, _ = self(inputs)

        # Examine the performance meter whether shares results between distributed devices.
        # Results: Actually different performance_meter.
        # print("batch_idx {}, performance_meter: {}".format(batch_idx, id(self.performance_meter)))
        self.performance_meter.update({t: get_output(output[t], t) for t in self.eval_tasks}, targets)

    def on_validation_epoch_start(self) -> None:
        from lightning.eval.evaluate_utils import PerformanceMeter
        self.performance_meter = PerformanceMeter(self.dataset_name, self.eval_tasks)

    def on_train_epoch_start(self) -> None:
        schs = self.lr_schedulers()
        if schs is None:
            adjust_learning_rate(base_lr=self.optimizer_cfg["base_lr"],
                                optimizer=self.optimizers(),
                                epoch=self.current_epoch,
                                max_epochs=self.max_epoch)
        else:
            for sch in schs:
                sch.step()

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        return super().on_train_epoch_end(unused=unused)

    def validation_epoch_end(self, outputs) -> None:
        eval_results = self.performance_meter.get_score(verbose=True)
        if len(self.eval_tasks) > 1:
            mtl_results = calculate_multi_task_performance(eval_results, self.single_task_results)
            print("MTL improvement: {}".format(mtl_results))
            self.log("MTL_improve", mtl_results, sync_dist=True)
        reported = {}
        for k in eval_results.keys():
            for ik in eval_results[k].keys():
                reported[k + '/' + ik] = eval_results[k][ik]
        self.log_dict(reported, sync_dist=True)

    def get_param_groups(self):
        heads = []
        backbones = []
        balancer = []
        rest = []
        rest_k = []
        for k, v in self.named_parameters():
            if k.startswith('head'):
                if self.is_fix_head:
                    if k.split('.')[1] not in self.aux_tasks:
                        heads.append(v)
                else:
                    heads.append(v)
                continue
            if not self.fix_encoder and k.startswith('backbone'):
                backbones.append(v)
                continue
            if k.startswith('balancer'):
                balancer.append(v)
                continue
            rest.append(v)
            rest_k.append(k)
        if len(rest_k) > 0:
            print("Not use following parameter in Main Optimizer:", rest_k)

        pg = [
            {'params': backbones, 'type': 'backbone'},
            {'params': heads, 'type': 'head'},
        ]
        if self.balancer.need_outer_optimize:
            pg.append({'params': balancer, 'type': 'balancer'})
        return pg

    def configure_optimizers(self):
        optimizer_type = self.optimizer_cfg['type']
        param_groups = self.get_param_groups()
        if optimizer_type == 'ADAM':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.optimizer_cfg["base_lr"],
                betas=(0.9, 0.999),
                weight_decay=self.optimizer_cfg["weight_decay"],
            )
        elif optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.optimizer_cfg["base_lr"],
                momentum=self.optimizer_cfg["momentum"],
                # nesterov=self.optimizer_cfg["nesterov"],
                weight_decay=self.optimizer_cfg["weight_decay"]
            )
        else:
            raise NotImplementedError(
                "Not Support Optimizer type {}".format(optimizer_type))
        if 'use_scheduler' in self.optimizer_cfg:
            import torch.optim as optim
            return {
                'optimizer': optimizer,
                'lr_scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epoch)
            }
        else:
            return {
                'optimizer': optimizer,
            }
