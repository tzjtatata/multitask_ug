# -*- coding: utf-8 -*-
# @Time : 2021/3/28 下午4:09
# @Author : lyz
# @Email : sqlyz@hit.edu.cn
# @File : string_utils
# @Project : multitask


def get_first_part_in_key(k, comma='.'):
    parts = k.split(comma)
    return parts[0], comma.join(parts[1:])
