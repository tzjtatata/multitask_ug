import os
import h5py
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from lightning.data.utils import convert_PIL_to_numpy

import sys
sys.path.append("../..")
from torchlet.utils.file_io import PathManager
import random
from tqdm import tqdm


freq_count = [629643, 10230, 980583, 35346, 6116738, 667206, 19968, 32925, 59992, 739075, 13484, 1658, 225700, 449262, 123748, 8605388, 34491, 371, 1339, 21042, 178307, 27602203, 95313, 47554, 8761, 10096, 54292, 4433817, 81472, 7730, 57857, 16773249, 1677344, 3125, 3272434, 1630290, 154648, 18568620, 49136971, 302569937, 0]
KEYS = ['img', 'semseg', 'depth']
COLOR_MAP = {
    'NYU_41': [(250, 170, 160), (30, 190, 260), (190, 190, 170), (220, 110, 100), (30, 240, 10), (230, 30, 120),
               (70, 150, 100), (10, 120, 200), (150, 100, 220), (200, 240, 200), (0, 200, 130), (60, 200, 0),
               (140, 60, 0), (80, 100, 110), (140, 20, 170), (190, 160, 200), (20, 40, 40), (80, 20, 70),
               (40, 260, 50), (160, 240, 180), (160, 110, 80), (210, 140, 140), (220, 100, 150), (220, 20, 20),
               (80, 100, 0), (200, 120, 50), (170, 170, 80), (160, 260, 120), (260, 230, 60), (200, 10, 120),
               (120, 240, 10), (100, 40, 10), (20, 50, 150), (220, 220, 70), (100, 120, 0), (30, 30, 10),
               (90, 240, 170), (50, 180, 140), (150, 60, 240), (120, 30, 240), (230, 80, 180)],
}


class NYUv2Dataset:

    def __init__(self, root, num_classes):
        """
            We assume that the file structure like this:
            - root
                - nyu_depth_v2_labeled.mat
                - splits.mat
            In fact, I don't want to use splits.mat

            Especially, this class output numpy.array as label, depth and image
        """
        self.num_classes = num_classes
        self.root = root
        self.file = h5py.File(os.path.join(self.root, "nyu_depth_v2_labeled.mat"), "r")
        self.labels_file = scio.loadmat(os.path.join(self.root, "labels{}_full.mat").format(self.num_classes))
        splits = scio.loadmat(os.path.join(self.root, "splits.mat"))
        self.train, self.test = splits['trainNdxs'], splits['testNdxs']
        self.length = self.file['images'].shape[0]

    def close(self):
        self.file.close()

    def generate_datalist(self, mode='train'):
        assert mode in ['train', 'test']
        ret = []
        splits = self.train if mode == 'train' else self.test
        for rank in range(len(splits)):
            i = splits[rank][0]
            data_dict = self.get_item(i)
            ret.append(data_dict)
        return ret

    def get_item(self, i):
        # split range from 1 to 1449
        return self.__getitem__(i-1)

    def get_name(self, item):
        name = self.file['rawRgbFilenames'][0][item]
        obj = self.file[name]
        return self.h5_to_str(obj)

    def get_label(self, item):
        return self.labels_file['labels40'][item]

    def h5_to_str(self, obj):
        s = ''.join([chr(c[0]) for c in obj])
        return s

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        name = self.get_name(i)
        img = Image.fromarray(self.preprocess(self.file['images'][i]), mode='RGB')
        img = self.rotate(img)
        img = convert_PIL_to_numpy(img, 'RGB')
        depth = Image.fromarray(self.file['depths'][i], mode='F')
        depth = self.rotate(depth)
        depth = convert_PIL_to_numpy(depth, 'F')
        label = Image.fromarray(self.get_label(i))
        label = self.rotate(label)
        label = convert_PIL_to_numpy(label, 'L').squeeze(2)
        data_dict = {
            'file_name': name,
            'height': 480,
            'width': 640,
            'image_id': i,
            'img': img,
            'annotations': [],
            'sem_seg': label,
            'depth': depth
        }

        return data_dict

    @staticmethod
    def data_show(data, num_classes=13):
        fig = plt.figure("Labeled Dataset Sample", figsize=(12, 5))

        ax = fig.add_subplot(1, 3, 1)
        NYUv2Dataset.show(ax, data['img'], "Color")
        ax = fig.add_subplot(1, 3, 2)
        NYUv2Dataset.show(ax, data['depth'], "Depth")
        ax = fig.add_subplot(1, 3, 3)
        sem_seg = map_label_to_color_v2(data['sem_seg'], num_classes=num_classes)
        NYUv2Dataset.show(ax, sem_seg, "SemSeg")
        plt.show()

    def preprocess(self, mat):
        img = np.moveaxis(mat, 0, -1)
        return img

    def rotate(self, img: Image):
        return img.rotate(-90, expand=True)


def show(ax, img, title, cmap=None):
    ax.axis('off')
    ax.set_title(title)
    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)


def getNYUv2DataList(mode='train', data_root="../data/NYUv2", num_classes=13):
    nyu = NYUv2Dataset(data_root, num_classes=num_classes)
    assert mode in ['train', 'test'], "Only Support train or test mode to read NYUv2 Dataset."
    datalist = nyu.generate_datalist(mode)
    nyu.close()
    return datalist


def make_seperate_NYUv2(data_root='../data/NYUv2', num_classes=13):
    root = os.path.abspath(data_root)

    nyu = NYUv2Dataset(root, num_classes=num_classes)
    prefix = "labels{}".format(num_classes)
    train_path = os.path.join(root, prefix, 'train')
    test_path = os.path.join(root, prefix, 'test')
    PathManager(train_path, is_clean=True)
    PathManager(test_path, is_clean=True)

    train_list = nyu.generate_datalist('train')
    new_list = save_NYU_datalist(train_path, train_list)
    scio.savemat(os.path.join(train_path, 'list.mat'), new_list)

    test_list = nyu.generate_datalist('test')
    new_list = save_NYU_datalist(test_path, test_list)
    scio.savemat(os.path.join(test_path, "list.mat"), new_list)


def save_NYU_datalist(target, data_list):
    names, imgs, depths, labels = [], [], [], []

    for data in tqdm(data_list):
        name = data['file_name']
        """
        e.g. office_0003/r-1294851609.500330-2975113445.ppm changed to office_0003-r-1294851609.500330-2975113445.png
        """
        # print(name)
        prefix = '-'.join(name.split('/'))[:-4]
        names.append(prefix)
        root = PathManager(os.path.join(target, prefix)).cur_path

        # print(prefix)
        img_path = os.path.join(root, prefix+".png")
        img = Image.fromarray(data['img'], mode='RGB')
        img.save(img_path)
        imgs.append(img_path)

        depth_path = os.path.join(root, prefix+"_depth.tiff")
        depth = Image.fromarray(data['depth'], mode='F')
        depth.save(depth_path)
        data['depth'] = depth_path
        depths.append(depth_path)

        # depth_read = Image.open(depth_path)
        # diff = convert_PIL_to_numpy(depth_read, format='F') - data['depth']
        # print("Shape: {}, Mean: {}, Total: {}".format(diff.shape, np.mean(diff), np.sum(diff)))

        semseg_path = os.path.join(root, prefix+"_semseg.png")
        semseg = Image.fromarray(data['sem_seg'], mode='L')
        semseg.save(semseg_path)
        labels.append(semseg_path)

        # semseg_read = Image.open(semseg_path)
        # diff = convert_PIL_to_numpy(semseg_read, format='L').squeeze(2) - data['sem_seg']
        # print("Shape: {}, Mean: {}, Total: {}".format(diff.shape, np.mean(diff), np.sum(diff)))

    return {
        'name': names,
        'img': imgs,
        'depth': depths,
        'sem_seg': labels
    }


def load_test():
    from time import perf_counter
    start = perf_counter()
    data = getNYUv2DataList()
    end = perf_counter()
    print(len(data))
    print("Used Time: {:.4f}s".format(end - start))


def map_NYU_to_few_class(root="../data/NYUv2/", is_output=True, num_classes=40):
    assert num_classes in [13, 40], "Not Support Other Number of Classes."
    origin = h5py.File(os.path.join(root, "nyu_depth_v2_labeled.mat"), "r")
    target_path = os.path.join(root, "labels{}_full.mat".format(num_classes))

    """
        origin has 894 classes, index from 0 to 894.
        But in mapping file, we only has 0 to 893, the 894 class is ignore class.
    """
    maps40 = scio.loadmat(os.path.join(root, "classMapping40.mat"))['mapClass'][0]
    if num_classes == 13:
        maps13 = scio.loadmat(os.path.join(root, "classMapping13.mat"))['mapClass'][0]
        maps = maps13
    else:
        maps13 = None
        maps = maps40

    map_statistic = [0] * num_classes
    for i in maps:
        map_statistic[i-1] += 1
    print(map_statistic)

    labels = origin['labels']
    new_labels = np.zeros(labels.shape, dtype=labels.dtype)
    origin_shape = labels.shape
    labels = np.array(labels).flatten()
    new_labels = new_labels.flatten()

    def map_chain(l):
        if num_classes == 13:
            return maps13[maps40[l-1]-1] - 1
        else:
            return maps40[l-1] - 1

    gap = 640 * 480
    data_statistic = [0]*(num_classes + 1)
    for i in range(len(labels)):
        if labels[i] == 0:
            new_labels[i] = num_classes
        else:
            new_label = map_chain(labels[i])
            new_labels[i] = new_label
            data_statistic[new_label] += 1
        if i != 0 and ((i+1) % gap == 0):
            print("{} / {}".format(i, len(labels)))
    new_labels = new_labels.reshape(origin_shape)
    print(data_statistic)
    print(new_labels.shape)
    if is_output:
        scio.savemat(target_path, {'labels40': new_labels})


def map_to_new_13label(root='../data/NYUv2'):
    """
    Map the mapping file class13Mapping.mat into my own format classMapping13.mat
    original class13Mapping.mat download from github/nyuv2-meta-data
    """
    old_map = scio.loadmat(os.path.join(root, "class13Mapping.mat"))
    scio.savemat(os.path.join(root, "classMapping13.mat"), {
        'mapClass': old_map['classMapping13'][0][0][0]
    })


def map_label_to_color(label, map_name='NYU_41'):
    new_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=int)
    cmap = COLOR_MAP[map_name]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            new_img[i, j] = cmap[label[i, j]]
    return new_img


class ColorMapper:

    def __init__(self, num_classes):
        self.num_classes = num_classes + 1
        self.lattice = math.floor(num_classes ** (1/3)) + 1
        self.gap = 255 // self.lattice
        self.start_point = (255 % self.gap) // 2
        print(self.gap)
        print(self.lattice)

    def get_color_map(self, c):
        color = [self.start_point] * 3
        tmp = c
        for i in range(3):
            color[i] += (tmp % self.lattice) * self.gap
            tmp = tmp // self.lattice
        return tuple(color)


def map_label_to_color_v2(label, num_classes=13):
    cmapper = ColorMapper(num_classes)
    new_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=int)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            new_img[i, j] = cmapper.get_color_map(label[i, j])
    return new_img


def median_frequency_balanced(data_list, num_classes=40):
    """
        This function is used to calculate dataset class balance weights.
    """
    class_count = np.zeros(num_classes + 1, dtype=np.int)
    for data in data_list:
        if isinstance(data['sem_seg'], str):
            sem_seg = convert_PIL_to_numpy(Image.open(data['sem_seg']), format='L').squeeze(2)
        else:
            sem_seg = data['sem_seg']
        labels = sem_seg.flatten()
        class_count += np.bincount(labels, minlength=num_classes+1).astype(np.int)
    print("class counting: {}".format(class_count))

    class_freqency = (class_count + 1.0) / np.sum(class_count)
    median = np.median(class_freqency)
    freq = 1.0 * median / class_freqency
    print("freq: {}".format(freq))
    return freq


def generate_class_to_color(num_classes):
    """
        This function split RGB space into multiple points.
        In order to Seperate Different class, each color margin with 10 in color space.
    """
    colors = []

    def get_int():
        return random.randint(0, 26)

    while len(colors) < num_classes:
        color = (get_int()*10, get_int()*10, get_int()*10)
        if color not in colors:
            colors.append(color)
    return colors


def NYUv2_data_show(ax, k, data):
    if k == 'img':
        show(ax, data, "Img")
    elif k == 'semseg':
        SemSeg = map_label_to_color_v2(data, num_classes=13)
        show(ax, SemSeg, "Semantic")
    elif k == 'depth':
        show(ax, data, 'Depth')