# -*- coding: utf-8 -*-
# @Time : 6/14/22 4:24 PM
# @Author : lyz
# @Email : sqlyz@hit.edu.cn
# @File : roi_head
# @Project : multitask
from detectron2.modeling.roi_heads import Res5ROIHeads
from multitask.model.heads.build import HEAD_REGISTRY

HEAD_REGISTRY.register(Res5ROIHeads)
