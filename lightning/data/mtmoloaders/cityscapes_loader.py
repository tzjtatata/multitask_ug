# Adapted from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py

import os

import torch
import time
import cv2
import numpy as np
import scipy.misc as m
from torchvision.transforms import transforms, InterpolationMode

from torch.utils import data

from lightning.data.mtmoloaders.loader_utils import recursive_glob
from lightning.data.mtmoloaders.segmentation_augmentations import *
from lightning.data.preloader import preloader

import matplotlib.pyplot as plt


cityscapes_augmentations = Compose([RandomRotate(10),
                                   RandomHorizontallyFlip()])


def read_bytes_images(file_name, annotations_base=None, depth_base=None):
    img_path = file_name
    lbl_path = os.path.join(annotations_base,
                            img_path.split(os.sep)[-2],
                            os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
    instance_path = os.path.join(annotations_base,
                                 img_path.split(os.sep)[-2],
                                 os.path.basename(img_path)[:-15] + 'gtFine_instanceIds.png')
    depth_path = os.path.join(depth_base,
                              img_path.split(os.sep)[-2],
                              os.path.basename(img_path)[:-15] + 'disparity.png')
    img = open(img_path, 'rb').read()
    lbl = open(lbl_path, 'rb').read()
    ins = open(instance_path, 'rb').read()
    depth = open(depth_path, 'rb').read()

    return img, lbl, ins, depth


class CITYSCAPES(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    colors = [#[  0,   0,   0],
              [128,  64, 128],
              [244,  35, 232],
              [ 70,  70,  70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170,  30],
              [220, 220,   0],
              [107, 142,  35],
              [152, 251, 152],
              [ 0, 130, 180],
              [220,  20,  60],
              [255,   0,   0],
              [  0,   0, 142],
              [  0,   0,  70],
              [  0,  60, 100],
              [  0,  80, 100],
              [  0,   0, 230],
              [119,  11,  32]]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, root, split=["train"], is_transform=False,
                 img_size=(512, 1024), augmentations=None):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.split_text = '+'.join(split)
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.transform_fn = transforms.Compose(
            [
                transforms.Resize((int(self.img_size[0] / 8), int(self.img_size[1] / 8)),
                                interpolation=InterpolationMode.NEAREST),
                # it would not be change image scales to [0, 255.0]
            ]
        )
        self.mean = np.array([123.675, 116.28, 103.53])
        # self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}

        self.files[self.split_text] = []
        for _split in self.split:
            self.images_base = os.path.join(self.root, 'leftImg8bit', _split)
            self.annotations_base = os.path.join(self.root, 'gtFine', _split)
            self.files[self.split_text] = recursive_glob(rootdir=self.images_base, suffix='.png')
            self.depth_base = os.path.join(self.root, 'disparity',  _split)

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.no_instances = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',\
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if len(self.files[self.split_text]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split_text, self.images_base))

        print("Found %d %s images" % (len(self.files[self.split_text]), self.split_text))

        # print("Prepare ins labels.")
        # from tqdm import tqdm
        # for i in tqdm(range(len(self))):
        #     img, lbl, ins, depth = self._read_item(i)
        #     img_path = self.files[self.split_text][i].rstrip()
        #     instance_path = os.path.join(self.annotations_base,
        #                              img_path.split(os.sep)[-2],
        #                              os.path.basename(img_path)[:-15] + 'gtFine_instance.npy')
        #     # if os.path.exists(instance_path):
        #         # continue
        #     lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        #     # Reference: https://segmentfault.com/a/1190000039409629
        #     ins_y, ins_x = self.encode_instancemap(lbl, ins)
        #     ins = np.stack((ins_y, ins_x)).astype(np.float32)
        #     np.save(instance_path, ins)
        #     # tick = time.perf_counter()
        #     # np.load(instance_path)
        #     # tock = time.perf_counter()
        #     # print("Load time: ", tock - tick)
        
        # all_times = 0.0
        # for i in tqdm(range(len(self))):
        #     img_path = self.files[self.split_text][i].rstrip()
        #     instance_path = os.path.join(self.annotations_base,
        #                              img_path.split(os.sep)[-2],
        #                              os.path.basename(img_path)[:-15] + 'gtFine_instance.npy')
        #     tick = time.perf_counter()
        #     np.load(instance_path)
        #     tock = time.perf_counter()
        #     all_times += (tock - tick)
        # print("Mean time:", all_times / len(self))

    def __len__(self):
        """__len__"""
        # return 100
        return len(self.files[self.split_text])

    def _read_item(self, index):
        img_path = self.files[self.split_text][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        instance_path = os.path.join(self.annotations_base,
                                     img_path.split(os.sep)[-2],
                                     os.path.basename(img_path)[:-15] + 'gtFine_instanceIds.png')
        depth_path = os.path.join(self.depth_base,
                                  img_path.split(os.sep)[-2],
                                  os.path.basename(img_path)[:-15] + 'disparity.png')
        img = m.imread(img_path)
        lbl = m.imread(lbl_path)
        ins = m.imread(instance_path)
        # Original MTMO ways to read depth, need mean and std to rescale depth map.
        # depth = np.array(m.imread(depth_path), dtype=np.float32)

        # Our way to read disparity correctly, follow cityscapes standard handler.
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # unchanged is -1
        depth[depth > 0.0] = (depth[depth > 0.0] - 1.) / 256.0  # output disparity

        return img, lbl, ins, depth

    def _read_item_from_ram(self, index):
        return None, None, None, None

    def __getitem__(self, index, is_preload=False):
        """__getitem__

        :param index:
        """
        # Note that we will not directly use CITYSCAPES, but use the CITYSCAPESWrapper instead.
        if is_preload:
            img, lbl, ins, depth = self._read_item_from_ram(index)
        else:
            img, lbl, ins, depth = self._read_item(index)

        # depth[depth!=0] = (depth[depth!=0] - self.DEPTH_MEAN[depth!=0]) / self.DEPTH_STD

        # tick = time.perf_counter()
        if self.augmentations is not None:
            img, lbl, ins, depth = self.augmentations(np.array(img, dtype=np.uint8), np.array(lbl, dtype=np.uint8), np.array(ins, dtype=np.int32), np.array(depth, dtype=np.float32))
        # tock = time.perf_counter()
        # print("Augmentation used time: {}".format(tock - tick))

        tick = time.perf_counter()
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        ins_y, ins_x = self.encode_instancemap(lbl, ins)
        # Zero-Mean, Std-Dev depth map
        tock = time.perf_counter()
        print("Generate instance used time: {}".format(tock - tick))

        # tick = time.perf_counter()
        if self.is_transform:
            img, lbl, ins, depth = self.transform(img, lbl, ins_y, ins_x, depth)
        # tock = time.perf_counter()
        # print("Transform used time: {}".format(tock - tick))

        return img, lbl, ins, depth

    def transform(self, img, lbl, ins_y, ins_x, depth):
        """transform

        :param img:
        :param lbl:
        :comments do mean for imgs and resize all images into size.
        """


        img = img[:, :, ::-1]  # BGR to RGB
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))

        # img_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]),
        #     transforms.Resize((self.img_size[0], self.img_size[1]))
        # ])

        classes = np.unique(lbl)
        # img = img_transform(img).float()
        img = transforms.ToTensor()(img).float()
        lbl = self.transform_fn(torch.from_numpy(lbl).unsqueeze(0)).long()
        ins = np.stack((ins_y, ins_x))
        ins = self.transform_fn(torch.from_numpy(ins)).float()
        depth = self.transform_fn(torch.from_numpy(depth).unsqueeze(0)).float()

        if not torch.all(torch.unique(lbl[lbl!=self.ignore_index]) < self.n_classes):
            print('after det', classes, torch.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        return img, lbl.squeeze(), ins, depth.squeeze()

    # it is slower than former. This is about 0.16, former is about 0.12
    # def transform(self, img, lbl, ins_y, ins_x, depth):
    #     """transform
    
    #     :param img:
    #     :param lbl:
    #     :comments do mean for imgs and resize all images into size.
    #     """
    #     img = img[:, :, ::-1]
    #     img = img.astype(np.float64)
    #     img -= self.mean
    #     img = m.imresize(img, (self.img_size[0], self.img_size[1]))
    #     # Resize scales images from 0 to 255, thus we need
    #     # to divide by 255.0
    #     img = img.astype(float) / 255.0
    #     # NHWC -> NCWH
    #     img = img.transpose(2, 0, 1)
    
    #     classes = np.unique(lbl)
    #     lbl = lbl.astype(float)
    #     lbl = m.imresize(lbl, (int(self.img_size[0]/8), int(self.img_size[1]/8)), 'nearest', mode='F') # TODO(ozan) /8 is quite hacky
    #     lbl = lbl.astype(int)
    
    #     ins_y = ins_y.astype(float)
    #     ins_y = m.imresize(ins_y, (int(self.img_size[0]/8), int(self.img_size[1]/8)), 'nearest', mode='F')
    
    #     ins_x = ins_x.astype(float)
    #     ins_x = m.imresize(ins_x, (int(self.img_size[0]/8), int(self.img_size[1]/8)), 'nearest', mode='F')
    
    #     depth = m.imresize(depth, (int(self.img_size[0]/8), int(self.img_size[1]/8)), 'nearest', mode='F')
    #     # depth = np.expand_dims(depth, axis=0)
    #     # if not np.all(classes == np.unique(lbl)):
    #     #    print("WARN: resizing labels yielded fewer classes")
    
    #     if not np.all(np.unique(lbl[lbl!=self.ignore_index]) < self.n_classes):
    #         print('after det', classes,  np.unique(lbl))
    #         raise ValueError("Segmentation map contained invalid class values")
    
    #     ins = np.stack((ins_y, ins_x))
    #     img = torch.from_numpy(img).float()
    #     lbl = torch.from_numpy(lbl).long()
    #     ins = torch.from_numpy(ins).float()
    #     depth = torch.from_numpy(depth).float()
    #     return img, lbl, ins, depth

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        # print('rgb : {}'.format(rgb.shape))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        #Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask==_voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask

    def encode_instancemap(self, mask, ins):
        # print(mask.shape, ins.shape, ins.max(), ins.min())
        ins[mask==self.ignore_index] = self.ignore_index
        for _no_instance in self.no_instances:
            ins[ins==_no_instance] = self.ignore_index
        # ins[ins==0] = self.ignore_index  #  Liyang Liu note that it is not needed.

        instance_ids = np.unique(ins)
        sh = ins.shape
        # print(sh)
        ymap, xmap = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing='ij')

        # out_ymap, out_xmap = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing='ij')  # Liyang Liu do not use it.
        out_ymap = np.ones(ymap.shape)*self.ignore_index
        out_xmap = np.ones(xmap.shape)*self.ignore_index

        for instance_id in instance_ids:
            if instance_id == self.ignore_index:
                continue
            instance_indicator = (ins == instance_id)
            # print(instance_id, np.sum(instance_indicator))
            coordinate_y, coordinate_x = np.mean(ymap[instance_indicator]), np.mean(xmap[instance_indicator])
            out_ymap[instance_indicator] = ymap[instance_indicator] - coordinate_y
            out_xmap[instance_indicator] = xmap[instance_indicator] - coordinate_x
        # print(out_ymap.min(), out_ymap.max())
        # print(out_xmap.min(), out_xmap.max())

        return out_ymap, out_xmap


class CITYSCAPESWrapper(CITYSCAPES):

    def __init__(self, root, target_type, split=["train"], is_transform=False,
                 img_size=(512, 1024), augmentations=None, is_preload=False):
        super(CITYSCAPESWrapper, self).__init__(root=root, split=split, is_transform=is_transform, img_size=img_size, augmentations=augmentations)
        print("Dataset is on {}".format(root))
        self.target_type = target_type
        self.is_preload = is_preload
        self._preload_bound = 750
        if self.is_preload:
            self._preload()

    def _read_item_from_ram(self, index):
        img, lbl, ins, depth = CITYSCAPESWrapper.decode_bytes_images(self.data_cache[index])
        return img, lbl, ins, depth

    @staticmethod
    def decode_bytes_images(bytes_images, use_depth=False, use_inverse_depth=False):
        from io import BytesIO
        b_img, b_lbl, b_ins, b_depth = bytes_images

        # tick = time.perf_counter()
        img = m.imread(BytesIO(b_img))
        lbl = m.imread(BytesIO(b_lbl))
        ins = m.imread(BytesIO(b_ins))
        # follwing is not quick than upper.
        # img = read_image(BytesIO(b_img))
        # lbl = read_image(BytesIO(b_lbl), format='L')
        # ins = read_image(BytesIO(b_ins))
        # tock = time.perf_counter()
        # print("Used time in m.imread(): {}".format(tock - tick))

        # tick = time.perf_counter()
        depth_array = np.asarray(bytearray(b_depth), dtype='uint8')
        depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED).astype(np.float32)  # unchanged is -1
        depth[depth > 0.0] = (depth[depth > 0.0] - 1.) / 256.0  # output disparity
        if use_depth:
            depth[depth > 0.0] = (0.209313 * 2262.52) / depth[depth > 0.0]
        if use_inverse_depth:
            depth[depth > 0.0] = depth[depth>0.0] / (0.209313 * 2262.52)
        # tock = time.perf_counter()
        # print("Used time in depth decode: {}".format(tock - tick))

        return img, lbl, ins, depth

    def _preload(self):
        from functools import partial
        imreader = partial(read_bytes_images, annotations_base=self.annotations_base, depth_base=self.depth_base)
        file_names = self.files[self.split_text]
        file_names = [
            i.rstrip()
            for i in file_names
        ]
        self.data_cache = preloader(self, imreader, file_names)

    def close(self):
        del self.data_cache

    def __getitem__(self, item, is_preload=False):
        imgs, labels, ins, depth = super(CITYSCAPESWrapper, self).__getitem__(item, is_preload=self.is_preload)
        targets = []
        for t in self.target_type:
            if t.lower() == 'semseg':
                targets.append(labels)
            if t.lower() == 'instance':
                targets.append(ins)
            if t.lower() == 'depth':
                targets.append(depth)
        return imgs, tuple(targets)


def get_depth_statistic_v1():
    """
    In this function, we calculate the mean and std of per images in training set.
    """
    local_path = '/home/lyz/PycharmProjects/multitask/data/cityscapes/'
    dst = CITYSCAPES(local_path, is_transform=False, augmentations=None)
    length = len(dst)

    from tqdm import tqdm
    std_collection = []
    mean_collection = []
    for i in tqdm(range(length)):
        _, _, _, depth = dst[i]
        mean_collection.append(np.mean(depth[depth!=0.0]))
        std_collection.append(np.std(depth[depth!=0.0]))

    print("mean: ", np.mean(mean_collection))
    print("std: ", np.mean(std_collection))
    print("mean_hist: ", np.histogram(mean_collection))
    print("std_hist: ", np.histogram(std_collection))


def visualize_dataset(is_preload=False):
    import matplotlib.pyplot as plt

    augmentations = Compose([RandomRotate(10),
                         RandomHorizontallyFlip()])

    local_path = '/home/lyz/PycharmProjects/multitask/data/cityscapes/'
    dst = CITYSCAPESWrapper(local_path, is_transform=True, augmentations=augmentations, is_preload=is_preload, target_type=['semseg', 'instance', 'depth'])

    f, axarr = plt.subplots(1, 5)
    imgs, targets = dst[1000]
    labels, instances, depth = targets
    imgs = imgs.numpy()[::-1, :, :]
    imgs = np.transpose(imgs, [1,2,0])
    axarr[0].imshow(imgs)
    print(imgs.shape)
    print(labels.shape)
    axarr[1].imshow(dst.decode_segmap(labels.numpy()))
    print(labels.shape)
    axarr[2].imshow(instances[0, :, :])
    print(instances[0, :, :].min(), instances[0, :, :].max())
    axarr[3].imshow(instances[1, :, :])
    print(instances[1, :, :].min(), instances[1, :, :].max())
    axarr[4].imshow(depth)
    plt.show()


def preloader_tester():
    augmentations = Compose([RandomRotate(10),
                             RandomHorizontallyFlip()])

    local_path = '/home/lyz/PycharmProjects/multitask/data/cityscapes/'
    dst = CITYSCAPES(local_path, is_transform=True, augmentations=augmentations)

    byte_images = preloader(dst)

    f, axarr = plt.subplots(1, 3)
    imgs, labels, instances, depth = CITYSCAPESWrapper.decode_bytes_images(byte_images[0])

    # imgs = imgs[::-1, :, :]
    # imgs = np.transpose(imgs, [1, 2, 0])
    print(imgs.shape)
    axarr[0].imshow(imgs)
    print(labels.shape)
    axarr[1].imshow(dst.decode_segmap(labels))
    # axarr[2].imshow(instances[0, :, :])
    # axarr[3].imshow(instances[1, :, :])
    axarr[2].imshow(depth)
    plt.show()


if __name__ == '__main__':
    # get_depth_statistic_v1()
    visualize_dataset(is_preload=True)
    # preloader_tester()

    # bs = 4
    # trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=True)
    # for i, data in enumerate(trainloader):
    #     imgs, labels, instances, depth = data
    #     imgs = imgs.numpy()[:, ::-1, :, :]
    #     imgs = np.transpose(imgs, [0,2,3,1])
    #
    #     f, axarr = plt.subplots(bs,5)
    #     for j in range(bs):
    #         axarr[j][0].imshow(imgs[j])
    #         axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
    #         axarr[j][2].imshow(instances[j,0,:,:])
    #         print(instances[j,0,:,:].min(), instances[j,0,:,:].max())
    #         axarr[j][3].imshow(instances[j,1,:,:])
    #         print(instances[j,1,:,:].min(), instances[j,1,:,:].max())
    #         axarr[j][4].imshow(depth[j,0])
    #     plt.show()
    #     break