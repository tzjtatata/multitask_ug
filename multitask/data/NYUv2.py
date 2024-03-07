import os
import h5py
import math
import hydra
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from lightning.data.utils import convert_PIL_to_numpy

import sys
sys.path.append("../..")
from torchlet.utils.file_io import PathManager
from time import perf_counter
import random
from tqdm import tqdm
import copy
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10


class NYUv2ImgDataset:

    def __init__(self, root, mode='train', num_classes=13):
        self.root = root
        self.num_classes = num_classes
        assert mode in ['train', 'test']
        self.mode = mode
        self.datas = scio.loadmat(os.path.join(self.root, "labels{}".format(num_classes), self.mode, 'list.mat'))
        self.length = len(self.datas['name'])

    def generate_datalist(self):
        data_dicts = []
        for i in range(self.length):
            data_dicts.append(self.__getitem__(i))
        return data_dicts

    def show_item(self, item):
        data = copy.deepcopy(self.__getitem__(item))
        print(data)
        data['img'] = convert_PIL_to_numpy(Image.open(data['img']), format='RGB')
        data['depth'] = convert_PIL_to_numpy(Image.open(data['depth']), format='F')
        data['sem_seg'] = convert_PIL_to_numpy(Image.open(data['sem_seg']), format='L').squeeze(2)
        print(np.bincount(data['sem_seg'].reshape(-1), minlength=self.num_classes))
        print(data['sem_seg'])
        NYUv2Dataset.data_show(data, self.num_classes)

    def __getitem__(self, i):
        data_dict = {
            'file_name': self.datas['name'][i].strip(),
            'height': 480,
            'width': 640,
            'image_id': i,
            'img': self.datas['img'][i].strip(),
            'annotations': [],
            'sem_seg': self.datas['sem_seg'][i].strip(),
            'depth': self.datas['depth'][i].strip()
        }

        return data_dict





class ETAEstimator:

    def __init__(self, total, gap):
        self.tick_time = 0
        self.origin_start = 0.0
        self.total = total
        self.gap = gap

    def tick(self):
        self.origin_start = perf_counter()
        self.tick_time += 1

    def tock(self, i):
        end = perf_counter()
        consumed_time = end - self.origin_start
        mean_time = consumed_time / self.tick_time
        eta = mean_time * ((self.total - i) * 1.0 / self.gap)
        self.tick_time += 1
        return eta, mean_time





if __name__ == "__main__":
    # map_to_new_13label()
    # map_NYU_to_few_class(num_classes=13)
    # map_NYU_to_few_class(num_classes=40)
    # load_test()
    # print(generate_class_to_color(41))

    # num_classes = 40
    # trainlist = getNYUv2DataList(num_classes=num_classes)
    # median_frequency_balanced(trainlist, num_classes=num_classes)
    # evallist = getNYUv2DataList(mode='test', num_classes=num_classes)
    # median_frequency_balanced(evallist, num_classes=num_classes)

    # make_seperate_NYUv2(num_classes=40)
    # make_seperate_NYUv2(num_classes=13)

    # nyu = NYUv2ImgDataset(root='../data/NYUv2')
    # nyu.show_item(100)
    #
    nyu40 = NYUv2ImgDataset(root='../data/NYUv2', mode='train', num_classes=40)
    nyu40.show_item(100)
    l = nyu40.generate_datalist()
    median_frequency_balanced(l, num_classes=40)

    #
    # label13 = scio.loadmat('../data/NYUv2/class13Mapping.mat')
    # maps13 = label13['classMapping13'][0][0]
    # label40 = scio.loadmat('../data/NYUv2/classMapping40.mat')
    # print(label13)
    # print(label40)


    # labelfile_13 = scio.loadmat('../data/NYUv2/labels13_full.mat')['labels40']
    # labelfile_40 = scio.loadmat('../data/NYUv2/labels40_full.mat')['labels40']
    # fig = plt.figure("Labeled Dataset Sample", figsize=(12, 5))
    #
    # item = 100
    # ax = fig.add_subplot(1, 2, 1)
    # data = labelfile_13[item]
    # label = Image.fromarray(data)
    # label = label.rotate(-90, expand=True)
    # label = convert_PIL_to_numpy(label, 'L').squeeze(2)
    # label = map_label_to_color_v2(label, 13)
    # NYUv2Dataset.show(ax, label, "Label13")
    #
    # ax = fig.add_subplot(1, 2, 2)
    # data = labelfile_40[item]
    # label = Image.fromarray(data)
    # label = label.rotate(-90, expand=True)
    # label = convert_PIL_to_numpy(label, 'L').squeeze(2)
    # label = map_label_to_color_v2(label, 40)
    # NYUv2Dataset.show(ax, label, "Label40")
    # plt.show()

    # nyu = NYUv2ImgDataset(root='../data/NYUv2', mode='train', num_classes=13)
    # data_list = nyu.generate_datalist()
    # print(len(data_list))