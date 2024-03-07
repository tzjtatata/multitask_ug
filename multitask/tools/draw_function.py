# -*- coding: utf-8 -*-
"""
    @Time    : 2021/3/10 1:36 上午
    @Author  : liyuanze
    @Email   : sqlyz@hit.edu.cn
    @File    : draw_function.py
    @Software: PyCharm
    @Comment :
    Input function which give x and output y, output a figure which shows this function.
"""
import os
from functools import partial
from matplotlib import pyplot as plt


def draw_function(
        func,
        freq=100,
        x_range=(0, 1),
        label='new figure',
        root='.',
        postprocess=True
):
    x_range = (x_range[0] * 1.0,
               x_range[1] * 1.0)

    step = (x_range[-1] - x_range[0]) * 1.0 / (freq+1)
    x_list = [x_range[0]+step*i for i in range(1, freq+2)]
    y_list = []
    for x in x_list:
        y_list.append(func(x))

    plt.plot(x_list, y_list, label=label)
    if postprocess:
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(x_range[0], x_range[1])
        plt.legend()


    # file_name = '_'.join(label.split(' ')).strip()
    # path = os.path.join(os.path.abspath(root), file_name)
    # return fig


def draw_2d_function(
        func,
        x_freq=100,
        y_freq=100,
        x_range=(0, 1),
        y_range=(0, 1),
        label='theta={}',
        root='.',
):
    y_range = (y_range[0] * 1.0,
               y_range[1] * 1.0)

    step = (y_range[-1] - y_range[0]) * 1.0 / (y_freq + 1)
    y_list = [y_range[0] + step * i for i in range(1, y_freq + 2)]
    for y in y_list:
        y_func = partial(func, y=y)
        draw_function(y_func, x_freq, x_range, label=label.format(y), postprocess=False)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(x_range[0], x_range[1])
    plt.legend()


def test_func(x, y, eps=1e-6):
    import math
    x = x + eps
    param1 = 0.5 / (y ** 2)
    param2 = math.log(y)
    return param1 * x + param2


def test_func2(x, y):
    import math
    param1 = 0.5 / (x ** 2)
    param2 = math.log(x)
    return param1 * y + param2


if __name__ == '__main__':
    func = partial(test_func, y=2.0)

    # draw_function(
    #     func,
    #     x_range=(0, 100),
    #     freq=1000
    # )
    # draw_2d_function(
    #     test_func,
    #     y_freq=10,
    #     x_range=(0, 1),
    #     y_range=(0, 2),
    # )
    draw_2d_function(
        test_func2,
        y_freq=10,
        x_range=(0, 0.25),
        y_range=(0, 1),
        label='loss={}'
    )
    plt.show()

