# -*- coding: utf-8 -*-
"""
  @Time : 2021/1/6 下午12:25
  @Author : lyz
  @Email : sqlyz@hit.edu.cn
  @File : MultiTaskNet.py
  @Project : multitask
"""
import numpy as np
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from lightning.eval.evaluate_utils import calculate_multi_task_performance

from lightning.model.regist_meta_arch import META_ARCH_REGISTRY
from multitask.model.backbones import build_backbone
from multitask.model.heads import build_heads
from lightning.utils.configs import configurable
from lightning.utils.utils import get_output
import torch.nn.functional as F
from lightning.balancer import build_balancer
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from multitask.model.backbones.resnet_vanden import BasicBlock


class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r 
        self.squeeze = nn.Sequential(nn.Linear(channels, channels//self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels//self.r, channels),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2,3))).view(B,C,1,1)
        return torch.mul(x, squeeze)


class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)


class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """
    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}
        
        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)


    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' %(a)]) for a in self.auxilary_tasks if a!= t} for t in self.tasks}
        out = {t: x['features_%s' %(t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0) for t in self.tasks}
        return out


class InitialTaskPredictionModule(nn.Module):
    """ Module to make the inital task predictions """
    def __init__(self, auxilary_tasks, input_channels, task_channels, output_channels: Dict):
        super(InitialTaskPredictionModule, self).__init__()        
        self.auxilary_tasks = auxilary_tasks

        # Per task feature refinement + decoding
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict({task: nn.Sequential(BasicBlock(channels, channels), BasicBlock(channels, channels)) for task in self.auxilary_tasks})
        
        else:
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(nn.Conv2d(input_channels, task_channels, 1, bias=False), 
                                nn.BatchNorm2d(task_channels))
                refinement[t] = nn.Sequential(BasicBlock(input_channels, task_channels, downsample=downsample),
                                                BasicBlock(task_channels, task_channels))
            self.refinement = nn.ModuleDict(refinement)

        self.decoders = nn.ModuleDict({task: nn.Conv2d(task_channels, output_channels[task], 1) for task in self.auxilary_tasks})


    def forward(self, features_curr_scale, features_prev_scale=None):
        if features_prev_scale is not None: # Concat features that were propagated from previous scale
            x = {t: torch.cat((features_curr_scale, F.interpolate(features_prev_scale[t], scale_factor=2, mode='bilinear')), 1) for t in self.auxilary_tasks}

        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}

        # Refinement + Decoding
        out = {}
        for t in self.auxilary_tasks:
            out['features_%s' %(t)] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out['features_%s' %(t)])

        return out


class FPM(nn.Module):
    """ Feature Propagation Module """
    def __init__(self, auxilary_tasks, per_task_channels):
        super(FPM, self).__init__()
        # General
        self.auxilary_tasks = auxilary_tasks
        self.N = len(self.auxilary_tasks)
        self.per_task_channels = per_task_channels
        self.shared_channels = int(self.N*per_task_channels)
        
        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//4, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//4))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//4, downsample=downsample),
                                     BasicBlock(self.shared_channels//4, self.shared_channels//4),
                                     nn.Conv2d(self.shared_channels//4, self.shared_channels, 1))

        # Dimensionality reduction 
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False),
                                    nn.BatchNorm2d(self.per_task_channels))
        self.dimensionality_reduction = BasicBlock(self.shared_channels, self.per_task_channels,
                                                    downsample=downsample)

        # SEBlock
        self.se = nn.ModuleDict({task: SEBlock(self.per_task_channels) for task in self.auxilary_tasks})

    def forward(self, x):
        # Get shared representation
        concat = torch.cat([x['features_%s' %(task)] for task in self.auxilary_tasks], 1)
        B, C, H, W = concat.size()
        shared = self.non_linear(concat)
        mask = F.softmax(shared.view(B, C//self.N, self.N, H, W), dim = 2) # Per task attention mask
        shared = torch.mul(mask, concat.view(B, C//self.N, self.N, H, W)).view(B,-1, H, W)
        
        # Perform dimensionality reduction 
        shared = self.dimensionality_reduction(shared)

        # Per task squeeze-and-excitation
        out = {}
        for task in self.auxilary_tasks:
            out[task] = self.se[task](shared) + x['features_%s' %(task)]
        
        return out


class MTINet(nn.Module):
    """ 
        MTI-Net implementation based on HRNet backbone 
        https://arxiv.org/pdf/2001.06902.pdf
    """
    def __init__(self, backbone, backbone_channels, heads, task_names):
        super(MTINet, self).__init__()
        # General
        self.tasks = task_names
        # 我们假设所有辅助任务都需要Refine
        self.auxilary_tasks = task_names
        self.num_scales = len(backbone_channels)
        self.channels = backbone_channels        

        # Backbone
        self.backbone = backbone
        
        # Feature Propagation Module
        self.fpm_scale_3 = FPM(self.auxilary_tasks, self.channels[3])
        self.fpm_scale_2 = FPM(self.auxilary_tasks, self.channels[2])
        self.fpm_scale_1 = FPM(self.auxilary_tasks, self.channels[1])

        # Initial task predictions at multiple scales
        output_channels = {
            'depth': 1,
            'semseg': 40
        }
        self.scale_0 = InitialTaskPredictionModule(self.auxilary_tasks, self.channels[0] + self.channels[1], self.channels[0], output_channels)
        self.scale_1 = InitialTaskPredictionModule(self.auxilary_tasks, self.channels[1] + self.channels[2], self.channels[1], output_channels)
        self.scale_2 = InitialTaskPredictionModule(self.auxilary_tasks, self.channels[2] + self.channels[3], self.channels[2], output_channels)
        self.scale_3 = InitialTaskPredictionModule(self.auxilary_tasks, self.channels[3], self.channels[3], output_channels)

        # Distillation at multiple scales
        self.distillation_scale_0 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[0])
        self.distillation_scale_1 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[1])
        self.distillation_scale_2 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[2])
        self.distillation_scale_3 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[3])
        
        # Feature aggregation through HRNet heads
        self.heads = nn.ModuleDict(heads)
        

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        # Backbone 
        x = self.backbone(x)
        
        # Predictions at multiple scales
            # Scale 3 
        x_3 = self.scale_3(x[3])
        x_3_fpm = self.fpm_scale_3(x_3)
            # Scale 2
        x_2 = self.scale_2(x[2], x_3_fpm)
        x_2_fpm = self.fpm_scale_2(x_2)
            # Scale 1
        x_1 = self.scale_1(x[1], x_2_fpm)
        x_1_fpm = self.fpm_scale_1(x_1)
            # Scale 0
        x_0 = self.scale_0(x[0], x_1_fpm)
        
        out['deep_supervision'] = {'scale_0': x_0, 'scale_1': x_1, 'scale_2': x_2, 'scale_3': x_3}        

        # Distillation + Output
        features_0 = self.distillation_scale_0(x_0)
        features_1 = self.distillation_scale_1(x_1)
        features_2 = self.distillation_scale_2(x_2)
        features_3 = self.distillation_scale_3(x_3)
        multi_scale_features = {t: [features_0[t], features_1[t], features_2[t], features_3[t]] for t in self.tasks}

        # Feature aggregation
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](multi_scale_features[t]), img_size, mode = 'bilinear')
            
        return out


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


@META_ARCH_REGISTRY.register()
class MTINetwork(pl.LightningModule):

    @configurable
    def __init__(
        self,
        backbone,
        task_names,
        balancer,
        heads,
        dataset_name,
        max_epoch,
        optimizer_cfg,
        single_task_results,
        inspect_grad_similarity
    ):
        super().__init__()
        # 这个数字是HRNet18特殊的
        self.mti = MTINet(backbone, [18, 36, 72, 144], heads, task_names=task_names)
        self.task_names = task_names
        self.balancer = balancer
        self.cal_kpi = False
        self.cal_grad = False
        self.automatic_optimization = False
        self.max_epoch = max_epoch
        self.optimizer_cfg = optimizer_cfg
        self.dataset_name = dataset_name
        self.single_task_results = single_task_results
        self.inspect_grad_similarity = inspect_grad_similarity
        self.grad_inspector = None

    @classmethod
    def from_config(cls, cfg):
        task_names = [t.lower() for t in cfg.dataset.task.tasks]
        single_task_results = dict(cfg.dataset.meta_data.single_task_results)
        return {
            "task_names": task_names,
            "backbone": build_backbone(cfg),
            "balancer": build_balancer(cfg.balancer.type, cfg),
            "heads": build_heads(cfg, task_names),
            "dataset_name": cfg.dataset.name,
            "max_epoch": cfg.dataset.solver.max_epoch,
            "optimizer_cfg": dict(cfg.dataset.solver.optimizer),
            "single_task_results": single_task_results,
            "inspect_grad_similarity": cfg.LTH.inspect_grad_similarity
        }

    def forward(self, x):
        ret = self.mti(x)
        return ret

    def training_step(self, batch, batch_idx):
        images = batch['image']
        targets = {task: batch[task] for task in self.task_names}
        output = self(images)

        losses = []
        for t in self.task_names:
            losses.append(self.mti.heads[t].losses(output[t], targets[t]))
            self.log("{}_loss".format(t),
                     losses[-1].item(), prog_bar=True, sync_dist=True)

        if self.inspect_grad_similarity:
            self.grad_inspector.step(losses)
        loss = self.balancer.run(losses)
        self.log("total_loss", loss.item(), sync_dist=True, prog_bar=True)
        opt = self.optimizers()
        # Backward
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log_dict(self.balancer.get_weights(), sync_dist=True)
        return loss

    def manual_backward(self, loss: Tensor, optimizer: Optional[Optimizer] = None, *args, **kwargs) -> None:
        loss.backward()

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs = batch['image']
        targets = {task: batch[task] for task in self.task_names}
        output = self(inputs)

        self.performance_meter.update({t: get_output(output[t], t) for t in self.task_names}, targets)

    def on_validation_epoch_start(self) -> None:
        from lightning.eval.evaluate_utils import PerformanceMeter
        self.performance_meter = PerformanceMeter(self.dataset_name, self.task_names)

    def on_train_epoch_start(self) -> None:
        adjust_learning_rate(base_lr=self.optimizer_cfg["base_lr"],
                             optimizer=self.optimizers(),
                             epoch=self.current_epoch,
                             max_epochs=self.max_epoch)

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        if self.inspect_grad_similarity:
            tb = self.logger.experiment
            tb.add_histogram('cosine_similarity', self.grad_inspector.summary(), self.current_epoch)
            self.grad_inspector.reset()
        return super().on_train_epoch_end(unused=unused)

    def validation_epoch_end(self, outputs) -> None:
        eval_results = self.performance_meter.get_score(verbose = True)
        if len(self.task_names) > 1:
            mtl_results = calculate_multi_task_performance(eval_results, self.single_task_results)
            print("MTL improvement: {}".format(mtl_results))
            self.log("MTL_improve", mtl_results, sync_dist=True)
        reported = {}
        for k in eval_results.keys():
            for ik in eval_results[k].keys():
                reported[k+'/'+ik] = eval_results[k][ik]
        self.log_dict(reported, sync_dist=True)

    def get_param_groups(self):
        mtan_w = []
        rest = []
        rest_k = []
        for k, v in self.named_parameters():
            if k.startswith('mti'):
                mtan_w.append(v)
                continue
            rest.append(v)
            rest_k.append(k)
        if len(rest_k) > 0:
            print("Not use following parameter in Main Optimizer:", rest_k)

        return [
            {'params': mtan_w, 'type': 'backbone'},
            {'params': rest, 'type': 'balancer'}
        ]

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
        if self.inspect_grad_similarity:
            self.grad_inspector = GradInspector(optimizer)
        return {
            'optimizer': optimizer,
        }


