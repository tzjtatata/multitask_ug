import torch
from torch import nn
from torch.nn import functional as F
from lightning.utils.configs import configurable

from multitask.model.heads.build import HEAD_REGISTRY
from .base import BasicHead
from .mtmo_head import SegmentationDecoder


def pixelL1Loss(output, gt):
    batch_size = output.shape[0]
    loss_matrix = F.l1_loss(output, gt, reduction='none').reshape(batch_size, -1)
    return torch.mean(torch.mean(loss_matrix, dim=1), dim=0)


def pixelRMSE(output, gt):
    batch_size = output.shape[0]
    loss_matrix = F.mse_loss(output, gt, reduction='none').reshape(batch_size, -1)
    return torch.mean(torch.sqrt(torch.mean(loss_matrix, dim=1)), dim=0)


def oldRMSE(output, gt):
    return torch.sqrt(F.mse_loss(output, gt, reduction='mean'))


class DepthHead(BasicHead):

    @configurable
    def __init__(
        self, 
        criterion,
        **kwargs
    ):
        super(DepthHead, self).__init__(**kwargs)
        self.criterion = criterion
    
    @classmethod
    def from_config(cls, cfg, task_name):
        ret = super().from_config(cfg, task_name)
        if "criterion" in cfg.dataset.model[task_name]:
            criterion_type = cfg.dataset.model[task_name].criterion
            if criterion_type == "L1":
                criterion = pixelL1Loss
                print("Depth Head Using pixel L1 Loss.")
            else:
                criterion = pixelRMSE
        else:
            criterion = oldRMSE
        ret['criterion'] = criterion

        return ret

    def forward(self, data):
        return self.map_func(data)

    def losses(self, pred, gt):
        """
        params:
            pred: input of current head, out from self.forward()
            gt: ground truth of current task.
            reduction: the reduction mode of return loss.
        """
        batch_size = pred.shape[0]
        output = pred.reshape(batch_size, -1)
        gt = gt.reshape(batch_size, -1)
        return self.criterion(output, gt)

    def kpi(self, pred, gt, eps=1e-6):
        # we use abs relative difference as api which belong to [0, 1]
        # Reference: Depth Map Prediction from a Single Image using a Multi-Scale Deep Network.
        output = pred.view(-1)
        gt = gt.view(-1)

        # Relative difference will be minus when network is not trained! Cause output - gt > gt.
        return torch.mean(1.0 - (torch.abs(output - gt) / (gt + eps)))


@HEAD_REGISTRY.register()
class Depth1Conv(DepthHead):

    def build_head(self):
        return nn.Conv2d(64, 1, kernel_size=3, padding=1)


@HEAD_REGISTRY.register()
class Depth2Conv2(DepthHead):

    def build_head(self):
        net = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        return net


@HEAD_REGISTRY.register()
class Depth2Conv(DepthHead):

    def build_head(self):
        net = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=2)
        )
        return net


@HEAD_REGISTRY.register()
class DepthASPP(DepthHead):

    @configurable
    def __init__(self, *, hidden_size, aspp_dim, **kwargs):
        self.hidden_size = hidden_size
        self.aspp_dim = aspp_dim
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg, task_name):
        ret =  super().from_config(cfg, task_name)
        ret["hidden_size"] = cfg.dataset.model.backbone.head.hidden_size
        ret["aspp_dim"] = cfg.dataset.model.backbone.head.in_planes
        return ret

    def build_head(self):
        return ASPPDecoder(
            aspp_dim=self.aspp_dim,
            hidden_size=self.hidden_size,
            task_type='R',
            out_channels=1
        )


@HEAD_REGISTRY.register()
class DepthMTMO(DepthHead):

    @configurable
    def __init__(self, *, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg, task_name):
        ret =  super().from_config(cfg, task_name)
        ret["hidden_size"] = cfg.dataset.model.backbone.head.hidden_size
        return ret

    def build_head(self):
        return SegmentationDecoder(num_class=1, task_type='R', hidden_size=self.hidden_size)

