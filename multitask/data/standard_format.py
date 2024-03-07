# -*- coding: utf-8 -*-
"""
  @Time : 2021/1/10 上午11:34
  @Author : lyz
  @Email : sqlyz@hit.edu.cn
  @File : standard_format.py
  @Project : multitask
"""

def construct_data_dict(
    img,
    file_name="",
    height=32,
    width=32,
    image_id=0,
    annotations=None,
    target=-1,
    sem_seg=None,
    depth=None
):
    if annotations is None:
        annos = []
    else:
        annos = annotations
    data_dict = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'image_id': image_id,
        'img': img,
        'target': target,
        'annotations': annos,
        'sem_seg': sem_seg,
        'depth': depth
    }

    return data_dict