# Adapted from: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/models/models.py

import torch
import torch.nn as nn
import torchvision
from lightning.utils.configs import configurable
from multitask.model.backbones import resnet_mit as resnet
from torchlet.backbone import BACKBONE_REGISTRY


def _cfg_MTMOBackbone(cfg):
    return {
        'pretrained': cfg.dataset.model.backbone.pretrained,
        'out_feats': cfg.dataset.model.fpn_features
    }


@BACKBONE_REGISTRY.register()
@configurable(from_config=_cfg_MTMOBackbone)
def MTMOBackbone(pretrained, out_feats=['res5']):
    orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
    return ResnetDilated(orig_resnet, dilate_scale=8, output_feats=out_feats)


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class ResnetDilated(nn.Module):
    
    def __init__(self, orig_resnet, dilate_scale=8, output_feats=['res5']):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.output_feats = output_feats
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # print(x.shape)
        out = {}
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        out['res1'] = x

        x = self.layer1(x)
        out['res2'] = x 
        x = self.layer2(x)
        out['res3'] = x
        x = self.layer3(x)
        out['res4'] = x
        x = self.layer4(x)
        out['res5'] = x
        # print(x.shape)
        for k in list(out.keys()):
            if k not in self.output_feats:
                out.pop(k)
        if len(out) == 1:
            return out[self.output_feats[0]]
        return out

    def get_last_layer(self):
        block = self.layer4[-1]
        return block.conv3
