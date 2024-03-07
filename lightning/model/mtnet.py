# -*- coding: utf-8 -*-
"""
  @Time : 2021/1/6 下午12:25
  @Author : lyz
  @Email : sqlyz@hit.edu.cn
  @File : MultiTaskNet.py
  @Project : multitask
"""
from sched import scheduler
from typing import Dict, Optional

import pytorch_lightning as pl
import torch

from lightning.model.lr_scheduler import PolynomialLR
from lightning.model.regist_meta_arch import META_ARCH_REGISTRY
from multitask.model.backbones import build_backbone
from multitask.model.heads import build_heads
from lightning.utils.configs import configurable
from lightning.eval.measurements import build_measurement
from lightning.balancer import build_balancer
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer


class Recorder:

    def __init__(self,
                 volume=200):
        self.records = []
        self.volume = volume

    def update(self, value):
        assert len(
            self.records) <= self.volume, "Some bugs accur that records max than volume."
        if len(self.records) == self.volume:
            self.records = self.records[1:]
        self.records.append(value)

    @property
    def is_full(self) -> bool:
        if self.is_accumulate:
            return len(self.records) >= self.volume
        else:
            return True

    def get_results(self):
        raise NotImplementedError(
            "You should implements get_results before call.")


class ConflictRecorder(Recorder):

    def __init__(self,
                 volume: int = 200,
                 is_mean: bool = True,
                 is_accumulate: bool = False):
        super(ConflictRecorder, self).__init__(volume)
        self.is_mean = is_mean
        self.is_accumulate = is_accumulate

    def get_results(self):
        if len(self.records) == 0:
            return {}
        if self.is_accumulate:
            num_task = len(self.records[0])
            results = []
            summary_fn = torch.mean if self.is_mean else torch.sum
            for i in range(num_task):
                t_collection = []
                for item in self.records:
                    t_collection.append(item[i])
                # turn gradients from (10, 320, 320) into (self.volume, 10, 320, 320)
                # Then mean on dim 0, turn back to (10, 320, 320)
                results.append(summary_fn(
                    torch.stack(t_collection, dim=0), dim=0))
        else:
            results = self.records[-1]

        # Hard Code that we only use this class in NYUv2.
        total_amount = results[0].flatten().size(0)
        difference_rate = torch.sum(
            (results[0] * results[1]) < 0.0) / total_amount * 1.
        return {
            "difference_rate": difference_rate
        }


@META_ARCH_REGISTRY.register()
class MultiTaskNet(pl.LightningModule):

    @configurable
    def __init__(
        self,
        backbone,
        task_names,
        num_tasks,
        balancer,
        heads,
        measurements,
        dataset_name,
        max_epoch,
        optimizer_cfg,
        scheduler_cfg,
        single_task_results,
        sta_kwargs,
        show_batch=False,
        fpn_features=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.fpn_features = fpn_features
        self.task_names = task_names
        self.balancer = balancer
        self.cal_kpi = False
        self.cal_grad = False
        self.heads = nn.ModuleDict(heads)
        self.automatic_optimization = False
        self.measurement = measurements
        self.max_epoch = max_epoch
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.dataset_name = dataset_name
        self.show_batch = show_batch
        self.single_task_results = single_task_results
        self.num_tasks = num_tasks

        self.subset_size = sta_kwargs["subset_size"]
        use_sta = sta_kwargs['use_sta']
        sta_overlap = sta_kwargs['is_overlap']
        if use_sta:
            if self.subset_size != -1:
                print("Using Fixed Subset Size {} For STA.".format(self.subset_size))
                self.sta_switch = 3
            else:
                self.sta_switch = 1 if sta_overlap else 2
        else:
            self.sta_switch = 0

    @classmethod
    def from_config(cls, cfg):
        task_names = [t.lower() for t in cfg.dataset.task.tasks]
        num_tasks = 40 if cfg.dataset.name =='celeba' else len(task_names)

        show_batch = 'show_batch' in cfg.dataset.eval and cfg.dataset.eval.show_batch
        if len(task_names) > 1 and 'single_task_results' in cfg.dataset.meta_data:
            single_task_results = dict(cfg.dataset.meta_data.single_task_results)
        else:
            single_task_results = None
        return {
            "task_names": task_names,
            "num_tasks": num_tasks,
            "backbone": build_backbone(cfg),
            "balancer": build_balancer(cfg.balancer.type, cfg),
            "heads": build_heads(cfg),
            "measurements": build_measurement(cfg, task_names),
            "dataset_name": cfg.dataset.name,
            "max_epoch": cfg.dataset.solver.max_epoch,
            "optimizer_cfg": dict(cfg.dataset.solver.optimizer),
            "scheduler_cfg": dict(cfg.dataset.solver.scheduler),
            "show_batch": show_batch,
            "single_task_results": single_task_results,
            "sta_kwargs": dict(cfg.dataset.sta_kwargs),
            "fpn_features": None
        }

    def log_grad(self, tag, grad, log_func):
        v = log_func(grad)
        # If use two gpus, This gradient will be grad mean.
        self.log(tag, v.item(), sync_dist=True)

    @property
    def use_kpi(self):
        return self.cal_kpi or type(self.balancer).__name__ == 'DTP'

    @property
    def use_grad(self):
        return self.cal_grad or type(self.balancer).__name__ in ['GradNorm']

    def _forward_backbone(self, data):
        features = self.backbone(data)
        if isinstance(features, Dict):
            for k in list(features.keys()):
                if k not in self.fpn_features:
                    features.pop(k)
            if len(features) == 1:
                return features['res5']
            else:
                return features
        else:
            return features

    def _forward_heads(self, representation):
        preds = []
        for task_name in self.task_names:
            head = self.heads[task_name]
            pred = head(representation)
            preds.append(pred)
        return preds

    def forward(self, x):
        shared = self._forward_backbone(x)
        preds = self._forward_heads(shared)
        return preds, shared

    def on_train_start(self):
        sch = self.lr_schedulers()
        if sch is not None and isinstance(sch, PolynomialLR):
            sch.set_max_steps(self.max_epoch)
        return super().on_train_start()

    def on_train_epoch_start(self) -> None:
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        return super().on_train_epoch_start()
    
    def _forward_loss(self, pred, gt):
        from lightning.model.vanden import generate_task_mask, generate_mask
        losses = []
        bs = pred[0].shape[0]
        mask = None
        switch = 0 if bs == 1 else self.sta_switch

        if switch == 3:
            from lightning.model.vanden import generate_mask_with_subset_size, check_if_task_not_use
            subset_mask = generate_mask_with_subset_size(bs, len(self.task_names), self.subset_size)
            while not check_if_task_not_use(subset_mask):
                subset_mask = generate_mask_with_subset_size(bs, len(self.task_names), self.subset_size)
            # print(subset_mask)
            subset_mask = subset_mask.to(self.device)

        for i, t in enumerate(self.task_names):
            if switch == 0:
                losses.append(self.heads[t].losses(pred[i], gt[i]))
            elif switch == 1:
                t_mask = generate_task_mask(bs)
                losses.append(
                    self.heads[t].losses(pred[i][t_mask == 1], gt[i][t_mask == 1])
                )
            elif switch == 2:
                if mask is None:
                    mask = generate_mask(bs, max_range=len(self.task_names))
                losses.append(
                    self.heads[t].losses(pred[i][mask == i], gt[i][mask == i])
                )
            elif switch == 3:
                # print("Task {}:".format(t), pred[t][subset_mask[:, i] == 1.0].shape, gt[t][subset_mask[:, i] == 1.0].shape)
                losses.append(
                    self.heads[t].losses(pred[i][subset_mask[:, i] == 1.0], gt[i][subset_mask[:, i] == 1.0])
                )

        return losses

    def training_step(self, batch, batch_idx):
        if self.balancer.need_optimizer and self.balancer._optim is None:
            self.balancer.set_optim(self.optimizers())
        img, targets = batch
        preds, shared = self(img)
        # print(img.dtype, preds[0].dtype, preds[1].dtype) # To This step, data.dtype is float32
        # losses = [
        #     self.heads[h].losses(preds[i], targets[i])
        #     for i, h in enumerate(self.task_names)
        # ]
        losses = self._forward_loss(preds, targets)
        # print(losses[0].dtype, losses[1].dtype)
        addition_info = {}

        if self.use_grad:
            layer = self.backbone.get_last_layer()
            grads = [
                torch.autograd.grad(task_loss, layer.parameters(), retain_graph=True)[0].detach()
                for task_loss in losses
            ]
            addition_info['grads'] = grads

            # Default: log grad norm if the model need grads.
            # for i, t in enumerate(self.task_names):
            #     self.log_grad("GradNorm/{}".format(t), grads[i], cal_gradnorm)
        if len(self.task_names) > 1:
            for i, t_name in enumerate(self.task_names):
                self.log("{}_loss".format(t_name),
                        losses[i], prog_bar=True, sync_dist=True)
                if self.use_kpi:
                    self.log("{}_kpi".format(t_name), addition_info['kpi'][i])

        loss = self.balancer.run(losses, **addition_info)
        opt = self.optimizers()
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
        self.log("total_loss", loss, sync_dist=True, prog_bar=True)

        self.log_dict(self.balancer.get_weights(), prog_bar=False, sync_dist=True)
        return loss

    def manual_backward(self, loss: Tensor, *args, **kwargs) -> None:
        if self.balancer.is_imtl:
            self.balancer._set_grad(loss, kwargs['repre'])
            return

        if self.balancer.is_pcgrad:
            self.balancer.pc_backward(loss)
            return
        loss.backward()

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        img, targets = batch
        preds, _ = self(img)
        if self.show_batch:
            from lightning.data.NYUv2 import handle_batch
            pred_batch = (img, tuple(targets+preds))
            handle_batch(
                pred_batch, target_type=self.task_names+self.task_names)
        out = []
        for i, t in enumerate(self.task_names):
            out.append(self.measurement[t].step(preds[i], targets[i]))
        return out

    def validation_epoch_end(self, outputs) -> None:
        # output
        summary = outputs[0]
        step_num = len(outputs)
        for i in range(1, step_num):
            for j in range(len(summary)):
                summary[j] += outputs[i][j]
        mtl_results = {}
        for i, t in enumerate(self.task_names):
            ret = self.measurement[t].summary(summary[i])
            if len(self.task_names) > 1:
                mtl_results[t] = ret
            self.log_dict(ret, sync_dist=True)
        if self.single_task_results is not None:
            from lightning.eval.evaluate_utils import calculate_multi_task_performance
            # 更改mtl_results的格式
            eval_results = {}
            for k, v in mtl_results.items():
                eval_results[k] = {
                    nk.split('/')[-1]: nv for nk, nv in v.items()
                }
            mtl_improve = calculate_multi_task_performance(eval_results, self.single_task_results)
            print("MTL improvement: {}".format(mtl_improve))
            self.log("MTL_improve", mtl_improve, sync_dist=True)

    def get_param_groups(self):
        heads = []
        backbones = []
        balancer = []
        rest = []
        rest_k = []
        for k, v in self.named_parameters():
            if k.startswith('head'):
                heads.append(v)
                continue
            if k.startswith('backbone'):
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
                nesterov=self.optimizer_cfg["nesterov"],
                weight_decay=self.optimizer_cfg["weight_decay"]
            )
        else:
            raise NotImplementedError(
                "Not Support Optimizer type {}".format(optimizer_type))
        scheduler_type = self.scheduler_cfg["type"]
        if scheduler_type == 'none':
            return {
                'optimizer': optimizer
            }
        if scheduler_type == 'MultiStep':
            milestones = self.scheduler_cfg["milestones"]
            if isinstance(milestones, int):
                milestones = range(
                    0, self.max_epoch, milestones)
                print("milestones is {}".format(milestones))
            lr_scheduler = MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=self.scheduler_cfg["gamma"]
            )
        elif scheduler_type == 'polynomial':
            lr_scheduler = PolynomialLR(
                optimizer,
                power=self.scheduler_cfg["power"]
            )
        else:
            raise NotImplementedError(
                "Not Support Scheduler type {}".format(scheduler_type))
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler
            }
        }


