# -*- coding: utf-8 -*-
# @Time : 2020/9/24 下午2:51
# @Author : lyz
# @Email : sqlyz@hit.edu.cn
# @File : resnet
# @Project : multitask
import torch
from torchvision.models import (
    ResNet
)
from torchvision.models.resnet import (
    BasicBlock,
    Bottleneck,
    model_urls
)
from torchvision.models.utils import load_state_dict_from_url


class BackboneResNet(ResNet):

    def __init__(self, cfg, block, layers, **kwargs):
        self.cfg = cfg
        super().__init__(block, layers, **kwargs)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def get_last_layer(self):
        block = self.layer4[-1]
        if isinstance(block, BasicBlock):
            return block.conv2
        elif isinstance(block, Bottleneck):
            return block.conv3
        else:
            raise NotImplementedError("Not Implement get_last_layer For This type {}".format(str(block)))


def _resnet(arch, cfg, block, layers, pretrained=False, progress=True, **kwargs):
    model = BackboneResNet(cfg, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model


def resnet18(cfg, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _resnet('resnet18', cfg, BasicBlock, [2, 2, 2, 2], pretrained=cfg.backbone.pretrained, progress=progress,
                   **kwargs)


def resnet34(cfg, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', cfg, BasicBlock, [3, 4, 6, 3], pretrained=cfg.MODEL.BACKBONE.IS_PRETRAINED, progress=progress,
                   **kwargs)


def resnet50(cfg, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', cfg, Bottleneck, [3, 4, 6, 3], cfg.MODEL.BACKBONE.IS_PRETRAINED, progress,
                   **kwargs)


def resnet101(cfg, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', cfg, Bottleneck, [3, 4, 23, 3], cfg.MODEL.BACKBONE.IS_PRETRAINED, progress,
                   **kwargs)


def resnet152(cfg, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', cfg, Bottleneck, [3, 8, 36, 3], cfg.MODEL.BACKBONE.IS_PRETRAINED, progress,
                   **kwargs)


RESNET_SUP = [
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
]

if __name__ == '__main__':
    a = torch.randn(3, 3, 640, 480)
    model = resnet18(pretrained=True)
    b = model(a)
    print(b.shape)

