# -*- coding: utf-8 -*-
# @Time : 2021/5/25 下午8:04
# @Author : lyz
# @Email : sqlyz@hit.edu.cn
# @File : cityscapes
# @Project : multitask
import math

import h5py
import torch

import hydra
from tqdm import tqdm
import numpy as np
import os
from collections import namedtuple
import pytorch_lightning as pl
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg, iterable_to_str, extract_archive, check_integrity, download_file_from_google_drive
from functools import partial
from lightning.data.utils import loop_until_success
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import cv2
import PIL
from matplotlib import pyplot as plt
from lightning.utils.configs import configurable
from .regist_dataset import DATA_REGISTRY


GROUP4DICT = {
    "upper": [1,3,4,5,8,9,11,12,15,17,23,28,35],
    "middle": [7,19,27,29,30,34],
    "lower": [6,14,16,21,22,24,36,37,38],
    "whole": [0,2,10,13,18,20,25,26,31,32,33,39]
}

HEADER_S = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
ATTR_NAMES = HEADER_S.split()
assert len(ATTR_NAMES) == 40, "Attributes for CelebA should be 40, but now only {}".format(len(ATTR_NAMES))


def read_bytes_images_for_CelebA(file_name):
    img = open(file_name, 'rb').read()
    return img


class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``. If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "CelebA"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        # ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        # ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        # ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            is_shuffle: bool = False
    ) -> None:
        import pandas
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        # identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        # bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        self.attr_names = list(attr.columns)

        mask = slice(None) if split_ is None else (splits[1] == split_)

        if is_shuffle:
            filename, landmarks_align, attr = splits[mask].index.values, landmarks_align[mask].values, attr[mask].values
            zipped = list(zip(list(filename), list(landmarks_align), list(attr)))
            np.random.shuffle(zipped)
            self.filename, self.landmarks_align, self.attr = zip(*zipped)
            self.landmarks_align = torch.as_tensor(self.landmarks_align)
            self.attr = torch.as_tensor(self.attr)
        else:
            self.filename = splits[mask].index.values
            self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
            self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}

        # This scales use to recompute resized(80, 80) landmark
        if self.transform is not None:
            h, w = self._find_resize_transform()
            if h is not None:
                self.scales = torch.tensor([h / 178.0, w / 218.0] * 5)
            else:
                self.scales = None
        else:
            self.scales = None
        # This scales turn landmarks coords from pixels to ratio.
        # self.scales = torch.tensor([1.0/218, 1.0 /178]*5)

        self.imgs = []
        self.preload()
        # self.is_preload = is_preload  # if on_server, we should preload dataset into RAM.
        # if self.is_preload:
        #     self.preload()

    def preload(self):
        from lightning.data.preloader import preloader
        filenames = [
            os.path.join(self.root, self.base_folder, 'img_align_celeba_png', f.split('.')[0] + ".png")
            for f in self.filename
        ]
        imreader = read_bytes_images_for_CelebA
        self.imgs = preloader(self, imreader, filenames[:500])

    def close(self):
        del self.imgs
        self.imgs = None

    @staticmethod
    def decode_byte_string(byte_string):
        X = cv2.imdecode(np.asarray(bytearray(byte_string), dtype='uint8'), cv2.IMREAD_COLOR)
        X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
        return X

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                print("{} MD5 Wrong".format(fpath))
                return False

        # Should check a hash of the images
        print(os.path.join(self.root, self.base_folder, "img_align_celeba_png"))
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba_png"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def _find_resize_transform(self):
        if not isinstance(self.transform, transforms.Compose):
            if isinstance(self.transform, transforms.Resize):
                return self.transform.size
            else:
                return None, None
        for t in self.transform.transforms:
            if isinstance(t, transforms.Resize):
                return t.size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # if not self.is_preload:
        #     filename = self.filename[index].split('.')[0] + ".png"
        #     X = cv2.imread(os.path.join(self.root, self.base_folder, "img_align_celeba_png", filename), cv2.IMREAD_COLOR)
        #     X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
        # else:
        #     X = self.imgs[index]
        X = CelebA.decode_byte_string(self.imgs[index])

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :] * 1.0)
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :] * self.scales)
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return 500
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class CelebABin(VisionDataset):

    ATTR_NAMES = ATTR_NAMES
    
    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_shuffle: bool = False,
    ) -> None:
        super(CelebABin, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.split = split
        self.is_shuffle = is_shuffle
        self.read_meta_data()

        # 这里是为了SA做准备，让多个Dataset的instance具有不同的数据排序。
        # 正常来说不使用。
        self.mapping = [i for i in range(self.length)]
        if self.is_shuffle:
            np.random.shuffle(self.mapping)

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')
        if self.transform is not None:
            h, w = self._find_resize_transform()
            if h is not None:
                self.scales = torch.tensor([h / 178.0, w / 218.0] * 5)
            else:
                self.scales = None
        else:
            self.scales = None

        self.datas = None
        self.preload()
    
    def set_bucket(self, new_buckets):
        self.mapping = new_buckets

    def preload(self):
        self.datas = []
        with open(os.path.join(self.root, "celeba_{}.bin".format(self.split)), 'rb') as self.data_fp:
            for i in tqdm(range(len(self))):
                self.datas.append(self._read_img(i))

    def close(self):
        del self.datas

    def remap(self, index : int):
        return self.mapping[index]

    def read_meta_data(self):
        # Read Meta Data
        with open(os.path.join(self.root, "celeba_{}_meta.bin".format(self.split)), 'rb') as meta_fp:
            meta_data_b = meta_fp.read()
            meta_data = np.frombuffer(meta_data_b, dtype=np.uint32)
            self.length, self.img_l, self.attr_l, self.lm_l = list(meta_data)
        self.total_l = self.img_l + self.attr_l + self.lm_l

    def _find_resize_transform(self):
        if not isinstance(self.transform, transforms.Compose):
            if isinstance(self.transform, transforms.Resize):
                return self.transform.size
            else:
                return None, None
        for t in self.transform.transforms:
            if isinstance(t, transforms.Resize):
                return t.size

    @loop_until_success
    def _read_img(self, index: int):
        origin_bytes = read_byte(self.data_fp, index, self.total_l)
        X = np.frombuffer(origin_bytes[:self.img_l], dtype=np.uint8).reshape(218, 178, 3).copy()

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'attr':
                target = np.frombuffer(origin_bytes[self.img_l: self.img_l + self.attr_l], dtype=np.float32).copy()
            else:
                target = np.frombuffer(origin_bytes[self.img_l + self.attr_l:], dtype=np.float32).copy()
            target = torch.as_tensor(target)
            if self.scales is not None and t == 'landmarks':
                target = target * self.scales
            targets.append(target)

        return X, tuple(targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, targets = self.datas[self.remap(index)]

        if self.transform is not None:
            X = self.transform(X)

        return X, targets

    def __len__(self) -> int:
        # return 500
        return len(self.mapping)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class CelebA8Group(CelebABin):
    """
        Set target_type = ['attr']
    """
    
    def preload(self):
        self.target_type = ['attr']
        super(CelebA8Group, self).preload()

    @staticmethod
    def generate_mask_by_list(l):
        mask = [
            False if i not in l else True
            for i in range(40)
        ]
        mask = torch.as_tensor(mask, dtype=bool)
        return mask

    def get_masks(self):
        self.masks = [
            CelebA8Group.generate_mask_by_list(v)
            for k, v in GROUP4DICT.items()
        ]

    def get_labels(self, labels):
        """
            labels is torch.Tensor which shape is (1, 40)
        """
        if not hasattr(self, "masks"):
            self.get_masks()
        targets = []
        for mask in self.masks:
            targets.append(labels.masked_select(mask))
        return targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, targets = self.datas[self.remap(index)]

        if self.transform is not None:
            X = self.transform(X)

        return X, self.get_labels(targets[0])


@DATA_REGISTRY.register()
class CelebADataModule(pl.LightningDataModule):

    @configurable
    def __init__(
        self,
        data_dir,
        target_type,
        resolution,
        batch_size,
        num_workers
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target_type = target_type
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augments = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resolution[0], resolution[1])),
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, self.resolution[0], self.resolution[1])
    
    @classmethod
    def from_config(cls, cfg):
        on_server = cfg.dataset.dataloader.on_server
        if on_server:
            data_dir = "/tmp_data/NYUv2"
        else:
            data_dir = os.environ.get("DETECTRON2_DATASETS")
            if data_dir is None:
                try:
                    data_dir = os.path.join(hydra.utils.get_original_cwd(), "data")
                except:
                    data_dir = os.path.abspath('./data')
        
        target_type = [t.lower() for t in cfg.dataset.task.tasks]
        # 这里是为了以后的部分标签做准备
        # target_type = []
        # if 'attr' in tasks:
        #     target_type += ATTR_NAMES
        # target_type += [t for t in tasks if t != 'attr']

        resolution = cfg.dataset.meta_data.in_resolution
        return {
            "data_dir": data_dir,
            "target_type": target_type,
            "resolution": resolution,
            "batch_size": cfg.dataset.dataloader.batch_size // cfg.dataset.gpus,
            "num_workers": cfg.dataset.dataloader.num_workers
        }

    def setup(self, stage: Optional[str] = None):

        # Assign train/val dataset for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_type(self.data_dir, split='train', target_type=self.target_type, transform=self.augments)
            # self.val_dataset = self.train_dataset
            self.val_dataset = self.dataset_type(self.data_dir, split='valid',  target_type=self.target_type, transform=self.augments)


        # if stage == 'test' or stage is None:
        #     self.nyuv2_test = NYUv2(self.data_dir, split='test', transform=self.test_transform, out_transform=self.out_transform)

    @property
    def dataset_type(self):
        # return CelebA
        if len(self.target_type) == 4:
            return CelebA8Group
        return CelebABin

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()

    def train_dataloader(self):
        batch_size = self.batch_size
        # if self.cfg.train_setting.trainer == 'SA-Interp':
        #     ratio = self.cfg.train_setting.sa_ratio
        #     task_num = len(self.target_type)
        #     batch_size = batch_size + (task_num - 1) * math.floor(batch_size * (1. - ratio))
        print("Batch Size is {}.".format(batch_size))
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            # pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class CelebAMSDataModule(CelebADataModule):

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = {t: self.dataset_type(self.data_dir, split='train', transform=self.augments,
                                                   target_type=t, is_shuffle=(t != self.target_type[0]))
                            for t in self.target_type
                            }
        self.val_dataset = self.dataset_type(self.data_dir, split='valid', transform=self.augments, target_type=self.target_type)

    def train_dataloader(self):
        batch_size = self.cfg.train_setting.batch_size // len(self.target_type)
        loaders = {
            t: DataLoader(
                self.train_dataset[t],
                batch_size=batch_size,
                num_workers=self.cfg.train_setting.num_workers,
                shuffle=True,
                drop_last=True,
            )
            for t in self.target_type
        }
        return loaders


class CelebADataSAModule(CelebADataModule):

    def train_dataloader(self):
        from lightning.data.sampler import SAInterpSampler
        sampler = SAInterpSampler(
            data_source=self.train_dataset,
            batch_size=self.cfg.train_setting.batch_size,
            drop_last=True,
            num_tasks=len(self.target_type),
            sa_ratio=self.cfg.train_setting.sa_ratio
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.cfg.train_setting.num_workers
        )


def show_celeba_img(img, landmarks, save_path=None):
    X = transforms.ToPILImage()(img)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(X)
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.detach().cpu().numpy()
    for i in range(5):
        x, y = landmarks[i * 2] * 80, landmarks[i * 2 + 1] * 80
        draw.rectangle((x - 2, y - 2, x + 2, y + 2), (255, 0, 0))
    if save_path is not None:
        X.save(save_path)
        return
    X.show()


def save_celebA_hdf5(split='train'):
    dataset = CelebA(
        root="/home/lyz/PycharmProjects/multitask/data/",
        split=split,
        target_type=['attr', 'landmarks']
    )
    # show_img(dataset[0])
    length = dataset.__len__()
    import h5py
    from tqdm import tqdm
    with h5py.File("/home/lyz/PycharmProjects/multitask/data/celeba_{}.hdf5".format(split), 'w') as f:
        imgs = f.create_dataset("images", shape=(length, 218, 178, 3), dtype='uint8')
        attr = f.create_dataset("attr", shape=(length, 40))
        landmarks = f.create_dataset("landmarks", shape=(length, 10))
        for i in tqdm(range(dataset.__len__())):
            img, targets = dataset._read_img(i)
            imgs[i] = img.astype(np.uint8)
            attr[i] = targets[0]
            landmarks[i] = targets[1]


def test_CelebA():
    t = transforms.Compose([
        # transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Resize((80, 80))
    ])
    dataset = CelebA(root="/home/lyz/PycharmProjects/multitask/data/", split='train', target_type=['attr', 'landmarks'],
                     transform=t, is_shuffle=True)
    img, target = dataset[1000]
    print(img.shape)
    X = transforms.ToPILImage()(img)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(X)
    for i in range(5):
        x, y = target[1][i * 2], target[1][i * 2 + 1]
        draw.rectangle((x - 2, y - 2, x + 2, y + 2), (255, 0, 0))
    X.show()
    print(target)
    print(target[0].shape)


def show_celeba_sample(img: torch.Tensor, target: Tuple[torch.Tensor, torch.Tensor]):
    X = transforms.ToPILImage()(img)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(X)
    for i in range(5):
        x, y = target[1][i * 2], target[1][i * 2 + 1]
        draw.rectangle((x - 2, y - 2, x + 2, y + 2), (255, 0, 0))
    X.show()
    print(target[0])


def save_celebA_as_bin(root="/home/lyz/PycharmProjects/multitask/data/", split='train'):
    img_fp = open(os.path.join(root, "celeba_{}.bin".format(split)), 'wb')
    dataset = CelebA(root="/home/lyz/PycharmProjects/multitask/data/", split=split, target_type=['attr', 'landmarks'])

    # Write Meta Data
    with open(os.path.join(root, "celeba_{}_meta.bin".format(split)), 'wb') as meta_fp:
        length = len(dataset)
        img, targets = dataset._read_img(0)
        img_l, attr_l, lm_l = len(img.tobytes()), len(targets[0].numpy().tobytes()), len(targets[1].numpy().tobytes())
        meta_array = np.array([length, img_l, attr_l, lm_l], dtype=np.uint32)
        meta_fp.write(meta_array.tobytes())

    # Write data
    for i in tqdm(range(length)):
        imgs, targets = dataset._read_img(i)
        img_b, attr_b, lm_b = imgs.tobytes(), targets[0].numpy().tobytes(), targets[1].numpy().tobytes()
        total_b = img_b+attr_b+lm_b
        img_fp.write(img_b)
        img_fp.write(attr_b)
        img_fp.write(lm_b)


def test_bin(root="/home/lyz/PycharmProjects/multitask/data/", split='train'):
    img_fp = open(os.path.join(root, "celeba_{}.bin".format(split)), 'rb')

    # Read Meta Data
    with open(os.path.join(root, "celeba_{}_meta.bin".format(split)), 'rb') as meta_fp:
        meta_data_b = meta_fp.read()
        meta_data = np.frombuffer(meta_data_b, dtype=np.uint32)
        length, img_l, attr_l, lm_l = list(meta_data)

    import time
    import random
    total_time = 0.0
    total_l = img_l + attr_l + lm_l
    for i in range(1000):
        pos = random.randint(1, 10000)
        t0 = time.perf_counter()
        origin_bytes = read_byte(img_fp, pos, total_l)
        img = np.frombuffer(origin_bytes[:img_l], dtype=np.uint8).reshape(218, 178, 3)
        attr = np.frombuffer(origin_bytes[img_l: img_l+attr_l], dtype=np.float32)
        lm = np.frombuffer(origin_bytes[img_l+attr_l:], dtype=np.float32)
        t1 = time.perf_counter()
        total_time += (t1 - t0)

    print("Total: {}, Mean: {}".format(total_time, total_time / 1000.))

    pos = random.randint(0, 10000)
    origin_bytes = read_byte(img_fp, pos, total_l)
    img = np.frombuffer(origin_bytes[:img_l], dtype=np.uint8).reshape(218, 178, 3)
    attr = np.frombuffer(origin_bytes[img_l: img_l + attr_l], dtype=np.float32)
    lm = np.frombuffer(origin_bytes[img_l + attr_l:], dtype=np.float32)
    print(img.shape, attr.shape, lm.shape)
    show_celeba_sample(torch.as_tensor(img.transpose(2, 0, 1)), (torch.as_tensor(attr), torch.as_tensor(lm)))


def read_byte(fp, pos, length):
    fp.seek(pos*length)
    return fp.read(length)


def test_CelebABin():
    t = transforms.Compose([
        # transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Resize((80, 80))
    ])
    dataset = CelebABin(root="/home/lyz/PycharmProjects/multitask/data/", split='train', target_type=['attr', 'landmarks'],
                     transform=t, is_shuffle=True)

    import time
    import random
    total_time = 0.0
    for i in range(1000):
        pos = random.randint(1, 10000)
        t0 = time.perf_counter()
        dataset[pos]
        t1 = time.perf_counter()
        total_time += (t1 - t0)
    print("Total: {}, Mean: {}".format(total_time, total_time / 1000.))

    img, target = dataset[1000]
    print(img.shape)
    X = transforms.ToPILImage()(img)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(X)
    for i in range(5):
        x, y = target[1][i * 2], target[1][i * 2 + 1]
        draw.rectangle((x - 2, y - 2, x + 2, y + 2), (255, 0, 0))
    X.show()
    print(target)
    print(target[0].shape)


def test_CelebA8Group():
    t = transforms.Compose([
        # transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Resize((80, 80))
    ])
    dataset = CelebA8Group(root="/home/lyz/PycharmProjects/multitask/data/", split='train', target_type=['attr'],
                     transform=t, is_shuffle=True)

    # import time
    # import random
    # total_time = 0.0
    # for i in range(1000):
    #     pos = random.randint(1, 10000)
    #     t0 = time.perf_counter()
    #     dataset[pos]
    #     t1 = time.perf_counter()
    #     total_time += (t1 - t0)
    # print("Total: {}, Mean: {}".format(total_time, total_time / 1000.))

    img, target = dataset[1000]
    print(img.shape)
    X = transforms.ToPILImage()(img)
    X.show()
    print(target)
    print(target[0].shape)


@hydra.main(config_path='/home/lyz/PycharmProjects/multitask/configs', config_name='celeba.yaml')
def test_MS(cfg):
    dataset = CelebAMSDataModule(cfg)
    dataset.data_dir = "/home/lyz/PycharmProjects/multitask/data"
    dataset.setup('fit')
    train_loaders = dataset.train_dataloader()
    for k in train_loaders.keys():
        img, _ = next(iter(train_loaders[k]))
        X = transforms.ToPILImage()(img[0])
        X.show()
        X = transforms.ToPILImage()(img[-1])
        X.show()


if __name__ == "__main__":
    split='valid'
    # test_CelebA()
    # save_celebA_as_bin(split=split)
    # test_bin(split=split)
    # test_CelebABin()
    test_CelebA8Group()
    # test_MS()

    # split = 'train'
    # save_celebA_hdf5(split)

    # dataset = CelebAHDF5(
    #     "/home/lyz/PycharmProjects/multitask/data/",
    #     split=split,
    #     target_type=["attr", "landmarks"],
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize((80, 80)),
    #     ])
    # )
    # img, targets = dataset[0]
    # X = transforms.ToPILImage()(img)
    # print(dataset.scales)
    # print(targets[0].shape, targets[1].shape)
    # print(targets[0])
    # print(targets[1])
    # from matplotlib import pyplot as plt
    # from PIL import ImageDraw
    #
    # draw = ImageDraw.Draw(X)
    # for i in range(5):
    #     x, y = targets[1][i * 2], targets[1][i * 2 + 1]
    #     draw.rectangle((x - 2, y - 2, x + 2, y + 2), (255, 0, 0))
    # X.show()
    #
    # import time
    # import random
    # start_time = time.time()
    # for i in range(1000):
    #     img, targets = dataset[random.randint(1, 10000)]
    # end_time = time.time()
    # print("mean time: {}".format((end_time - start_time) / 100.0))





