# -*- coding: utf-8 -*-
# @Time : 2020/9/24 下午3:35
# @Author : lyz
# @Email : sqlyz@hit.edu.cn
# @File : __init__.py
# @Project : multitask
from .deeplab_like import ASPP
from .mtmo_backbone import MTMOBackbone
from .VGG import AutoEncoderVGG16
from .build import build_backbone
from .resnet_dilated import ResnetVanden
from .hrnet import hrnet_w18
from .vgg16 import vgg16
