# -*- coding: utf-8 -*-
"""
  @Time : 2021/1/10 下午12:42
  @Author : lyz
  @Email : sqlyz@hit.edu.cn
  @File : classify.py
  @Project : multitask
"""
from random import sample
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .base import BasicHead
from .build import HEAD_REGISTRY
from lightning.utils.configs import configurable


@HEAD_REGISTRY.register()
class BinaryFC(BasicHead):

    def build_head(self):
        in_plane = self.cfg.dataset.model.backbone.in_planes[-1]
        return nn.Sequential(
            nn.Linear(in_plane, 1),
            nn.Sigmoid()
        )

    def losses(self, pred, gt):
        pred, gt = pred.flatten(), gt.flatten()
        return F.binary_cross_entropy(pred, gt)


@HEAD_REGISTRY.register()
class CPFC(BasicHead):

    @configurable
    def __init__(
            self,
            num_classes,
            dim,
            **kwargs
    ):
        self.num_classes = num_classes
        self.dim = dim
        super().__init__(**kwargs)
        self.criterion = nn.CrossEntropyLoss()

    @classmethod
    def from_config(cls, cfg, task_name):
        ret = super().from_config(cfg, task_name)
        ret['num_classes'] = cfg.dataset.meta_data[task_name].num_classes
        ret['dim'] = cfg.dataset.model.backbone.in_planes[-1]
        return ret

    def build_head(self):
        net = nn.Linear(self.dim, self.num_classes)
        return net

    def forward(self, data):
        return self.map_func(data)

    def losses(self, pred, gt):
        return self.criterion(pred, gt)


def check_celeba_mask(mask):
    for i in range(40):
        if torch.sum(mask == i) == 0:
            return False
    return True


@HEAD_REGISTRY.register()
class MultiLabel(CPFC):

    def build_head(self):
        # heads = {}
        # for t in self.attr_names:
        #     heads[t] = nn.Sequential(
        #         nn.Linear(self.dim, 1),
        #         nn.Sigmoid()
        #     )
        # return nn.ModuleDict(heads)
        return nn.Sequential(
            nn.Linear(self.dim, self.num_classes),
            nn.Sigmoid()
        )

    # def forward(self, data):
    #     for t in self.attr_names:
    #         self.map_func

    def losses(self, pred, gt, cl_mask=None):
        # We follow Multi-Task as Multi-Objective Learning settings. Use binary log_softmax+nll=sigmoid+bce
        # When evaluation, use torch.round (default threshold=0.5) to generate predict label.
        gt = gt * 1.0  # change to float
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        if cl_mask is not None:
            loss = loss.masked_select(cl_mask.to(loss.device) == 1.0)
        return torch.mean(loss)
        # TODO: 这里先暂时通过改代码来将这个变成STA。后面再改回来。以上是原来代码，以下为新代码
        # gt = gt * 1.0  # change to float
        # sample_loss = F.binary_cross_entropy(pred, gt, reduction='none')  # (bs, 40)
        # mask = torch.randint_like(sample_loss, low=0, high=2) * 1.0
        # return torch.mean(sample_loss * mask)
        # TODO: 这里先暂时通过改代码来将这个变成STA-Nonoverlap。后面再改回来。以上是原来代码，以下为新代码
        # gt = gt * 1.0  # change to float
        # sample_loss = F.binary_cross_entropy(pred, gt, reduction='none')  # (bs, 40)
        # bs = sample_loss.shape[0]
        # mask = torch.randint(low=0, high=40, size=(bs, ))
        # while not check_celeba_mask(mask):
        #     mask = torch.randint(low=0, high=40, size=(bs, ))
        # real_mask = torch.zeros_like(sample_loss)
        # for i in range(bs):
        #     real_mask[i, mask[i].item()] = 1.0
        # return torch.mean(sample_loss * real_mask)


def generate_mask_with_subset_size(bs, num_tasks, subset_size):
    # 生成过程：
    # 生成(bs, num_tasks)的基准tensor，其中每列有subset_size个1.
    base_v = torch.zeros(bs, num_tasks)
    base_v[:, :subset_size] = 1.0
    # 生成随机打乱的tensor, 作为取数的基准
    indexs = torch.stack([torch.randperm(num_tasks) for _ in range(bs)], dim=0)  # indexs: (bs, num_tasks)
    # 通过取数，生成最后的mask
    subset_mask = torch.gather(base_v, dim=1, index=indexs)
    return subset_mask


def generate_mask_with_subset_size_1(bs, num_tasks, subset_size):
    # subset_size=1
    # 生成过程：
    # 生成(bs, num_tasks)的基准tensor，其中每行有subset_size个1.
    base_v = torch.zeros(bs - num_tasks, num_tasks)
    base_v[:, :subset_size] = 1.0
    # 生成随机打乱的tensor, 作为取数的基准
    indexs = torch.stack([torch.randperm(num_tasks) for _ in range(bs - num_tasks)],
                         dim=0)  # indexs: (bs-num_tasks, num_tasks)
    # 通过取数，生成最后的mask
    # 为了保证每列都不为全0，前40行
    mask_sta = torch.eye(num_tasks)
    # 以防万一，虽然感觉像是和空气斗智斗勇。。
    mask_sta = mask_sta[torch.randperm(num_tasks)].clone().contiguous()
    mask_end = torch.gather(base_v, dim=1, index=indexs)
    subset_mask = torch.cat((mask_sta, mask_end), 0)
    return subset_mask


def check_if_task_not_use(mask):
    bs, num_tasks = mask.shape
    for i in range(num_tasks):
        if torch.sum(mask[:, i]) == 0:
            return False
    return True


@HEAD_REGISTRY.register()
class CelebAWrapper(nn.Module):

    @configurable
    def __init__(self, heads, task_name, sta_switch, use_cl, subset_size) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(heads)
        self.task_names = task_name
        self.sta_switch = sta_switch
        self.use_cl = use_cl
        self.subset_size = subset_size

    @classmethod
    def from_config(cls, cfg, heads, task_name):
        sta_switch = 0
        if 'sta_kwargs' in cfg.dataset:
            if cfg.dataset.sta_kwargs.use_sta:
                sta_switch = 1
                # Decrepcated.
                if cfg.dataset.sta_kwargs.is_overlap:
                    sta_switch = 2
                if cfg.dataset.sta_kwargs.subset_size != -1:
                    print("Using Fixed Subset Size {} For STA.".format(cfg.dataset.sta_kwargs.subset_size))
                    sta_switch = 3
        
        use_cl = cfg.CL.use_cl if 'CL' in cfg else False
        return {
            "heads": heads,
            "task_name": task_name,
            "sta_switch": sta_switch,
            "use_cl": use_cl,
            "subset_size": cfg.dataset.sta_kwargs.subset_size,
        }

    def forward(self, x):
        outs = []
        for t in self.task_names:
            _out = self.heads[t](x)  # _out: (bs, 1)
            outs.append(_out)
        return torch.cat(outs, dim=1)

    def losses(self, pred, gt, cl_mask=None):
        gt = gt * 1.0  # change to float
        loss = F.binary_cross_entropy(pred, gt, reduction='none')  # loss: (bs, 40)
        bs = loss.shape[0]
        if self.use_cl:
            assert cl_mask is not None
            loss = loss * cl_mask.to(loss.device)
        if bs != 1 and not self.use_cl and self.sta_switch == 3:
            if self.subset_size==1:
                subset_mask = generate_mask_with_subset_size_1(bs, len(self.task_names), self.subset_size)
            else:
                subset_mask = generate_mask_with_subset_size(bs, len(self.task_names), self.subset_size)
                while not check_if_task_not_use(subset_mask):
                    subset_mask = generate_mask_with_subset_size(bs, len(self.task_names), self.subset_size)
            # print(subset_mask)
            subset_mask = subset_mask.to(loss.device)
            loss = loss * subset_mask
        elif bs != 1 and not self.use_cl and self.sta_switch != 0:
            # TODO: We do not implement non-overlap here.
            mask = torch.randint_like(loss, low=0, high=2) * 1.0
            loss = loss * mask

        # print(loss[:10,:10])
        losses = {}
        for i, t in enumerate(self.task_names):
            losses[t] = torch.mean(loss[:, i])
        return losses


@HEAD_REGISTRY.register()
class SingleLabel(CPFC):

    def build_head(self):
        return nn.Sequential(
            nn.Linear(self.dim, 1),
            nn.Sigmoid()
        )


@HEAD_REGISTRY.register()
class MultiLabelMTL(CPFC):

    def build_head(self):
        heads = {}
        for t in range(self.num_classes):
            heads[str(t)] = nn.Linear(self.dim, 2)
        return nn.ModuleDict(heads)

    def forward(self, data):
        outs = []
        for t in range(self.num_classes):
            out = self.map_func[str(t)](data)
            # out: (BS, 2)
            out = F.softmax(out, dim=-1)
            # out: (BS, 2) -> (BS, 2)
            outs.append(out)
        # ret: (BS*num_classes, 2)
        if self.training:
            return torch.concat(outs, dim=0)
        else:
            return torch.max(torch.stack(outs, dim=1), dim=-1).values

    def losses(self, pred, gt):
        # We follow Multi-Task as Multi-Objective Learning settings. Use binary log_softmax+nll=sigmoid+bce
        # When evaluation, use torch.round (default threshold=0.5) to generate predict label.
        # gt = gt * 1.0  # change to float
        # gt: (BS, num_classes) -> (BS*num_classes, )
        gt = gt.flatten().long()
        return self.criterion(pred, gt)


@HEAD_REGISTRY.register()
class VGGClassifier(CPFC):

    def build_head(self):
        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )
        return classifier


class CPRouting(BasicHead):

    def __init__(self, cfg):
        self.num_classes = cfg.TASK.CLASSPREDICT.NUMCLASSES
        self.dimension = 128
        super(CPRouting, self).__init__(cfg)
        self.num_transforms = self.dimension - self.num_classes
        self.prototype = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(self.num_classes, self.num_transforms),
                    torch.eye(self.num_classes, self.num_classes)
                ],
                dim=1
            ),
            requires_grad=True
        )
        self.transforms = nn.Parameter(
            torch.eye(self.num_transforms, self.dimension)
        )

    def build_head(self):
        net = nn.Linear(512, self.dimension, bias=False)
        return net

    def forward(self, data):
        # print("CPR.forward data.shape:", data.shape)
        embed = self.map_func(data)  # embed: (B, D)
        # print("CPR.forward embed.shape:", embed.shape)
        # print(embed)
        broadcast_embed = embed.unsqueeze(1).repeat(1, self.num_classes, 1)  # broadcast_embed: (B, C, D)
        # print("CPR.forward broadcast_embed.shape:", broadcast_embed.shape)
        prototype = self._normalize(self.prototype)
        prototype_diff = broadcast_embed - prototype  # prototype_diff: (B, C, D)
        transforms = self._normalize(self.transforms)  # transforms: (T, D)
        coord = torch.matmul(prototype_diff, transforms.TL)  # coord: (B, C, T)
        intra_distance = torch.sum(torch.abs(coord), dim=-1)
        # print("Intra_distance", intra_distance)
        """
            There is two ways: 
            1. torch.sum(coord.unsqueeze(3).repeat(1, 1, 1, self.dimension) * transforms, dim=2)
            2. torch.matmul(coord, transforms)
            But there exists some difference. So We use first way.
        """
        # inter_diff: (B, C, D)
        reconstruct_diff = torch.sum(coord.unsqueeze(3).repeat(1, 1, 1, self.dimension) * transforms, dim=2)
        # inter_distance: (B, C)
        inter_distance = torch.norm(prototype_diff - reconstruct_diff, dim=-1)

        return -inter_distance

    def _normalize(self, v):
        # v: (C, D)
        v = v / torch.norm(v, dim=1, keepdim=True)
        return v

    def losses(self, pred, gt):
        """
            @gt: (B, )
            @pred: (B, C)
        """
        # print("CPR.losses gt.shape:", gt.shape)
        # print("CPR.losses pred.shape:", pred.shape)
        # gt = gt.unsqueeze(1)
        # onehot = torch.zeros_like(pred)
        # onehot.scatter_(1, gt, 1)
        # print(onehot[:3], gt[:3])
        # reverse_onehot = -2 * onehot + 1

        # return torch.mean(torch.sum(torch.exp(reverse_onehot * pred), dim=1), dim=0)
        # return torch.mean(torch.sum(reverse_onehot * pred, dim=1), dim=0)

        return nn.functional.cross_entropy(pred, gt)
