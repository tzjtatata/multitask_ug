# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International
import os
import skimage
import cv2

from PIL import Image
from typing import Optional
import numpy as np
import torch.utils.data as data
from lightning.data.mypath import MyPath


class NYUD(data.Dataset):
    """
    NYUD dataset 
    including more data which only has depth prediction.

    Data can also be found at:
    https://drive.google.com/file/d/14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw/view?usp=sharing

    """

    def __init__(self,
                root=MyPath.db_root_dir('nyuv2'),
                split='val',
                transform=None,
                retname=True,
                overfit=False,
                max_samples=-1
                ):

        self.root = root
        self._real_root = os.path.join(root, split)
        """
            目录结构如下：
            nyuv2/
                train/
                    bedroom_0136_out/
                        depth/
                            1.png
                            2.png
                            ...
                        rgb/
                            1.png
                            2.png
                            ...
                    bedroom_0074_out/
                    ...
                val/
                    ...
        """

        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []

        # Depth
        self.depths = []


        print('Initializing dataloader for NYUD {} set'.format(''.join(self.split)))
        # 读取Depth Only的数据集
        for scene in os.listdir(self._real_root):
            scene_path = os.path.join(self._real_root, scene)
            # 必须是一个文件夹，否则就跳过
            if not os.path.isdir(scene_path):
                continue

            # 存在只有image没有depth的情况
            # _scene_images = [os.path.join(scene_path, 'rgb', f_name) for f_name in os.listdir(os.path.join(scene_path, 'rgb')) if f_name.endswith('.png')]
            # _scene_depths = [os.path.join(scene_path, 'depth', f_name) for f_name in os.listdir(os.path.join(scene_path, 'depth')) if f_name.endswith('.png')]
            _fnames = [f_name for f_name in os.listdir(os.path.join(scene_path, 'rgb')) \
                        if f_name.endswith('.png') and \
                            os.path.exists(os.path.join(scene_path, 'depth', f_name))]
            _scene_images = [os.path.join(scene_path, 'rgb', f_name) for f_name in _fnames]
            _scene_depths = [os.path.join(scene_path, 'depth', f_name) for f_name in _fnames]

            # 为了避免这种情况，我们只使用有对应depth的images.
            _im_ids = [scene + "_" + f_name.split('.')[0] for f_name in os.listdir(os.path.join(scene_path, 'rgb')) if f_name.endswith('.png')]
            assert len(_scene_depths) == len(_scene_images), "Scene {} in path {} has different images and depths.".format(scene, scene_path)
            self.im_ids += _im_ids
            self.images += _scene_images
            self.depths += _scene_depths
        
        # 限制数据集的大小
        if max_samples != -1:
            self.images = self.images[:max_samples]
            self.depths = self.depths[:max_samples]
            self.im_ids = self.im_ids[:max_samples]

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        _depth = self._load_depth(index)
        if _depth.shape[:2] != _img.shape[:2]:
            print('RESHAPE DEPTH')
            _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {
                'image': str(self.im_ids[index]),
                'im_size': (_img.shape[0], _img.shape[1])
            }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_depth(self, index):
        # _depth = np.load(self.depths[index])
        # 这样直接读取的结果是：_depth是一个uint16位。范围是0~65535
        # print(self.depths[index])
        _depth = np.asarray(Image.open(self.depths[index]), np.uint16)
        # 我们的做法是：skimage.img_as_float()能将一个np.ndarray转化成float。包括binary,uint和int都可以转。
        # 具体到uint16转float32，实际操作是除以65535
        # 另外：我们发现这个原始图片和795张的那个depth，好像正好差了10倍，猜测是从0-10归一化到0-1，所以这里也乘以10，以对齐那边的尺度
        return skimage.img_as_float32(_depth) * 10.0

    def __str__(self):
        return 'NYUD Depth (split=' + str(self.split) + ')'


class NYUFull(data.Dataset):
    """
    NYUD dataset 
    including Official NYUv2(795 for train) and Raw Depth dataset(22W3000 for train)

    Data can also be found at:
    https://drive.google.com/file/d/14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw/view?usp=sharing

    """

    def __init__(self,
                split='val',
                transform=None,
                retname=True,
                overfit=False,
                max_samples=-1
                ):
        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.retname = retname

        print('Initializing dataloader for NYUD {} set'.format(''.join(self.split)))
        from lightning.data.nyud import NYU_MT
        self.official_dataset = NYU_MT(split=split, transform=self.transform, do_edge=False,
                            do_semseg=True,
                            do_normals=False,
                            do_depth=True, overfit=False)
        if split == 'val':
            self.raw_dataset = []
        else:
            self.raw_dataset = NYUD(split=split, transform=self.transform, overfit=False, max_samples=max_samples)

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self)))

    def __getitem__(self, index):
        if index < len(self.official_dataset):
            sample = self.official_dataset[index]
            sample['meta']['semseg'] = 1
            sample['meta']['depth'] = 1
        else:
            sample = self.raw_dataset[index - len(self.official_dataset)]
            sample['meta']['semseg'] = 0
            sample['meta']['depth'] = 1
        return sample

    def __len__(self):
        return len(self.official_dataset) + len(self.raw_dataset)

    def __str__(self):
        return 'NYUD Full (split=' + str(self.split) + ')'
