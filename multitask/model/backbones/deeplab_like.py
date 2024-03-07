import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlet.backbone import BACKBONE_REGISTRY


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# In UW paper, ASPP is used to generate sharing features, not in head.
@BACKBONE_REGISTRY.register()
class ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # self.project = nn.Sequential(
        #     nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        #     nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return res
        # return self.project(res)


class ASPPBackbone(nn.Module):

    def __init__(self, cfg, verbose=False):
        super(ASPPBackbone, self).__init__()
        from multitask.model.backbones.mtmo_backbone import MTMOBackbone
        self.verbose = verbose
        self.resnet = MTMOBackbone(cfg)
        self.ASPP = ASPP(in_channels=2048, atrous_rates=[12, 24, 36])

    def forward(self, x):
        out = self.resnet(x)
        if self.verbose:
            print(out.shape)
        out = self.ASPP(out)
        return out


if __name__ == '__main__':
    backbone = ASPPBackbone("", verbose=True)
    input_t = torch.randn((12, 3, 256, 512))
    out = backbone(input_t)
    print(out.shape)
