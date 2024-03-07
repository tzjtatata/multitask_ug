import torch
from torch import nn
from torch.nn import functional as F
from .base import BasicHead
from .mtmo_head import SegmentationDecoder
# from .aspp_head import ASPPDecoder
from lightning.utils.configs import configurable
from multitask.model.heads.build import HEAD_REGISTRY


class InstanceHead(BasicHead):

    def forward(self, data):
        return self.map_func(data)

    def losses(self, pred, gt):
        """
        params:
            pred: input of current head, out from self.forward()
            gt: ground truth of current task.
            reduction: the reduction mode of return loss.
        """
        mask = gt != 250
        lss = F.l1_loss(pred[mask], gt[mask], size_average=False)
        lss = lss / mask.data.sum()
        return lss

    def kpi(self, pred, gt, eps=1e-6):
        # we use abs relative difference as api which belong to [0, 1]
        # Reference: Depth Map Prediction from a Single Image using a Multi-Scale Deep Network.
        output = pred.view(-1)
        gt = gt.view(-1)

        # Relative difference will be minus when network is not trained! Cause output - gt > gt.
        return torch.mean(1.0 - (torch.abs(output - gt) / (gt + eps)))


class InstanceConv(InstanceHead):

    def build_head(self):
        net = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        return net


@HEAD_REGISTRY.register()
class InstanceMTMO(InstanceHead):

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
        return SegmentationDecoder(num_class=2, task_type='I', hidden_size=self.hidden_size)


# class InstanceASPP(InstanceHead):

#     def build_head(self):
#         return ASPPDecoder(
#             aspp_dim=self.cfg.backbone.head.in_planes,
#             hidden_size=self.cfg.backbone.head.hidden_size,
#             task_type='I',
#             out_channels=2
#         )

