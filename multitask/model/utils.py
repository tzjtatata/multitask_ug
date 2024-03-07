# -*- coding: utf-8 -*-
"""
  @Time : 2020/11/3 上午2:38
  @Author : lyz
  @Email : sqlyz@hit.edu.cn
  @File : utils.py
  @Project : multitask
"""

def show_models(state_dict):
    for k, v in state_dict.items():
        print(k, v.shape)