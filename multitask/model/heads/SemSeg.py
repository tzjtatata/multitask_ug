import torch
from torch import nn
from torch.nn import functional as F

from lightning.utils.configs import configurable
from multitask.model.heads.build import HEAD_REGISTRY
from .base import BasicHead
from torchlet.model.loss import FocalLoss
from .mtmo_head import SegmentationDecoder


# NYUv2 class weights.
class_weights = {
    "class40": [0.04602844, 0.10827641, 0.15144615, 0.28261854, 0.29355468, 0.42177805,
 0.49242567, 0.49618962, 0.45067839, 0.5180064,  0.49835783, 0.74285091,
 0.63500554, 1. ,        0.80136174, 1.00398477, 1.09060081, 1.2478062,
 0.93397707, 1.55416101, 1.18557406, 0.68572009, 1.85299349, 1.89399383,
 1.79614205, 2.7049257,  2.9785801,  2.45798844, 2.42262059, 3.77865027,
 3.2865949,  3.90517014, 3.50669353, 4.11366428, 3.68374636, 4.57697536,
 4.74510871, 0.4373809,  0.53112158, 0.21508723, 0.05638134],
    "class13": [0.98137291, 6.43438891, 2.38111454, 1.01934788, 0.35149407, 0.23603804,
 0.2676436,  1.73051234, 1.46459446, 1.14572779, 6.23697632, 0.14626262,
 0.72497894, 0.19578021]
}


class SemSegHead(BasicHead):

    @configurable
    def __init__(
        self,
        num_classes, 
        ignore_index,
        weight=None,
        **kwargs
    ):
        self.num_classes = num_classes
        super().__init__(**kwargs)
        
        if weight is not None:
            self.register_buffer('class_weight', torch.tensor(class_weights['class{}'.format(self.num_classes-1)], requires_grad=False))
        else:
            self.class_weight = None

        # ignore_index is -1 for cityscapes npy, 255 for cityscapes
        # TODO: Notice!! This changes to 250 because the mtmo setting. Our default setting is 255
        self.ignore_index = ignore_index
        self.criterion = F.cross_entropy
        # self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=self.reduction, ignore_index=self.ignore_index)
    
    @classmethod
    def from_config(cls, cfg, task_name):
        ret = super().from_config(cfg, task_name)
        num_classes = cfg.dataset.meta_data.semseg.num_classes 
        if cfg.dataset.name != 'cityscapes':
            num_classes += 1
        ret["num_classes"] = num_classes

        if cfg.dataset.name.startswith("nyuv2"):
            ret["weight"] = class_weights['class{}'.format(num_classes-1)]
        
        ret["ignore_index"] = 255 if 'MTMO' not in cfg.dataset.type else 250
        return ret

    def build_head(self):
        raise NotImplementedError("You Should Implement build_head function before using this head.")

    def forward(self, data):
        return self.map_func(data)

    def losses(self, pred, gt):
        # print(pred.shape, gt.shape)
        batch_size = pred.shape[0]
        output = self.flatten_pred(pred)
        gt = gt.view(-1)
        if self.reduction == 'none':
            return torch.mean(
                self.criterion(output, gt, weight=self.class_weight, reduction=self.reduction, ignore_index=self.ignore_index).reshape(batch_size, -1), dim=-1
            )
        return self.criterion(output, gt, weight=self.class_weight, reduction=self.reduction, ignore_index=self.ignore_index)

    def flatten_pred(self, pred):
        batch_size = pred.shape[0]
        output = pred.view(pred.size(0), pred.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        return output

    def kpi(self, logit, gt, eps=1e-6):
        pred = torch.argmax(self.flatten_pred(logit), dim=-1)
        gt = gt.view(-1)
        kpi = torch.sum(pred == gt) / (gt.shape[0] + eps)  # This metric is pAcc

        return kpi


@HEAD_REGISTRY.register()
class SemSeg1Conv(SemSegHead):

    def build_head(self):
        return nn.Conv2d(64, self.num_classes, kernel_size=3, padding=1)


@HEAD_REGISTRY.register()
class SemSeg2Conv2(SemSegHead):

    def build_head(self):
        return nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, self.num_classes, kernel_size=1)
        )


@HEAD_REGISTRY.register()
class SemSeg2Conv(SemSegHead):
    """
        This 2Conv version is for outputing 80x80 resolution output.
    """

    def build_head(self):
        net = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(64, self.num_classes, kernel_size=3, padding=1, stride=2)
        )
        return net


@HEAD_REGISTRY.register()
class SemSegMTMO(SemSegHead):

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
        return SegmentationDecoder(self.num_classes, task_type='C', hidden_size=self.hidden_size)


@HEAD_REGISTRY.register()
class SemSegASPP(SemSegHead):

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
            task_type='C',
            out_channels=self.num_classes
        )

