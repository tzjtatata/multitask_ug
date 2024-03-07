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


#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

""" 
    Implementation of MTAN  
    https://arxiv.org/abs/1803.10704 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from multitask.model.backbones.resnet_vanden import ResNet, conv1x1, Bottleneck
from multitask.model.backbones.resnet_dilated import ResnetDilated


class AttentionLayer(nn.Sequential):
    """ 
        Attention layer: Takes a feature representation as input and generates an attention mask 
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(AttentionLayer, self).__init__(
                    nn.Conv2d(in_channels=in_channels, 
                        out_channels=mid_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=mid_channels, 
                        out_channels=out_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(out_channels),
                    nn.Sigmoid())


class RefinementBlock(nn.Sequential):
    """
        Refinement block uses a single Bottleneck layer to refine the features after applying task-specific attention.
    """
    def __init__(self, in_channels, out_channels):
        downsample = nn.Sequential(conv1x1(in_channels, out_channels, stride=1),
                                    nn.BatchNorm2d(out_channels))
        super(RefinementBlock, self).__init__(Bottleneck(in_channels, out_channels//4, downsample=downsample))


class MTAN(nn.Module):
    """ 
        Implementation of MTAN  
        https://arxiv.org/abs/1803.10704 

        Note: The implementation is based on a ResNet backbone.
        
        Argument: 
            backbone: 
                nn.ModuleDict object which contains pre-trained task-specific backbones.
                {task: backbone for task in p.TASKS.NAMES}
        
            heads: 
                nn.ModuleDict object which contains the task-specific heads.
                {task: head for task in p.TASKS.NAMES}
        
            stages: 
                a list of the different stages in the network 
                ['layer1', 'layer2', 'layer3', 'layer4']
 
            channels: 
                dict which contains the number of channels in every stage
            
            downsample:
                dict which tells where to apply 2 x 2 downsampling in the model        

    """
    def __init__(self, backbone, heads: nn.ModuleDict, task_names: List,
                stages: list, channels: dict, downsample: dict): 
        super(MTAN, self).__init__()
        
        # Initiate model                  
        self.tasks = task_names
        assert(isinstance(backbone, ResNet) or isinstance(backbone, ResnetDilated))
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        assert(set(stages) == {'layer1','layer2','layer3','layer4'})
        self.stages = stages
        self.channels = channels

        # Task-specific attention modules
        self.attention_1 = nn.ModuleDict({task: AttentionLayer(channels['layer1'], channels['layer1']//4,
                                                        channels['layer1']) for task in self.tasks})
        self.attention_2 = nn.ModuleDict({task: AttentionLayer(2*channels['layer2'], channels['layer2']//4,
                                                        channels['layer2']) for task in self.tasks})
        self.attention_3 = nn.ModuleDict({task: AttentionLayer(2*channels['layer3'], channels['layer3']//4,
                                                        channels['layer3']) for task in self.tasks})
        self.attention_4 = nn.ModuleDict({task: AttentionLayer(2*channels['layer4'], channels['layer4']//4,
                                                        channels['layer4']) for task in self.tasks})

        # Shared refinement
        self.refine_1 = RefinementBlock(channels['layer1'], channels['layer2'])
        self.refine_2 = RefinementBlock(channels['layer2'], channels['layer3'])
        self.refine_3 = RefinementBlock(channels['layer3'], channels['layer4'])
        
        # Downsample
        self.downsample = {stage: nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity() for stage, downsample in downsample.items()}


    def forward(self, x):
        img_size = x.size()[-2:]
        
        # Shared backbone
        # In case of ResNet we apply attention over the last bottleneck in each block.
        # Other backbones can be included by implementing the forward_stage_except_last_block
        # and forward_stage_last_block
        u_1_b = self.backbone.forward_stage_except_last_block(x, 'layer1')
        u_1_t = self.backbone.forward_stage_last_block(u_1_b, 'layer1')  

        u_2_b = self.backbone.forward_stage_except_last_block(u_1_t, 'layer2')
        u_2_t = self.backbone.forward_stage_last_block(u_2_b, 'layer2')  
        
        u_3_b = self.backbone.forward_stage_except_last_block(u_2_t, 'layer3')
        u_3_t = self.backbone.forward_stage_last_block(u_3_b, 'layer3')  
        
        u_4_b = self.backbone.forward_stage_except_last_block(u_3_t, 'layer4')
        u_4_t = self.backbone.forward_stage_last_block(u_4_b, 'layer4') 

        ## Apply attention over the first Resnet Block -> Over last bottleneck
        a_1_mask = {task: self.attention_1[task](u_1_b) for task in self.tasks}
        a_1 = {task: a_1_mask[task] * u_1_t for task in self.tasks}
        a_1 = {task: self.downsample['layer1'](self.refine_1(a_1[task])) for task in self.tasks}
        
        ## Apply attention over the second Resnet Block -> Over last bottleneck
        a_2_mask = {task: self.attention_2[task](torch.cat((u_2_b, a_1[task]), 1)) for task in self.tasks}
        a_2 = {task: a_2_mask[task] * u_2_t for task in self.tasks}
        a_2 = {task: self.downsample['layer2'](self.refine_2(a_2[task])) for task in self.tasks}
        
        ## Apply attention over the third Resnet Block -> Over last bottleneck
        a_3_mask = {task: self.attention_3[task](torch.cat((u_3_b, a_2[task]), 1)) for task in self.tasks}
        a_3 = {task: a_3_mask[task] * u_3_t for task in self.tasks}
        a_3 = {task: self.downsample['layer3'](self.refine_3(a_3[task])) for task in self.tasks}
        
        ## Apply attention over the last Resnet Block -> No more refinement since we have task-specific
        ## heads anyway. Testing with extra self.refin_4 did not result in any improvements btw.
        a_4_mask = {task: self.attention_4[task](torch.cat((u_4_b, a_3[task]), 1)) for task in self.tasks}
        a_4 = {task: a_4_mask[task] * u_4_t for task in self.tasks}

        # Task-specific heads
        out = {task: self.heads[task](a_4[task]) for task in self.tasks}
        out = {task: F.interpolate(out[task], img_size, mode='bilinear') for task in self.tasks} 
    
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
class MTANet(pl.LightningModule):

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
        mtan_args,
        inspect_grad_similarity
    ):
        super().__init__()
        self.mtan = MTAN(backbone, heads, task_names=task_names, **mtan_args)
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
            "inspect_grad_similarity": cfg.LTH.inspect_grad_similarity,
            "mtan_args": dict(cfg.dataset.model.mtan_kwargs)
        }

    def forward(self, x):
        ret = self.mtan(x)
        return ret

    def training_step(self, batch, batch_idx):
        images = batch['image']
        targets = {task: batch[task] for task in self.task_names}
        output = self(images)

        losses = []
        for t in self.task_names:
            losses.append(self.mtan.heads[t].losses(output[t], targets[t]))
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
            if k.startswith('mtan'):
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


