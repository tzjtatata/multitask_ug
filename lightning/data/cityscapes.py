import torch
import h5py

import json
import hydra
import numpy as np
import os
from collections import namedtuple
import pytorch_lightning as pl
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import cv2

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg, iterable_to_str, extract_archive
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from lightning.data import transforms as T
from lightning.data.transforms import Augmentation
from lightning.data.utils import read_image, convert_PIL_to_numpy
from lightning.data.preloader import preloader
from fvcore.transforms import Transform
from .regist_dataset import DATA_REGISTRY

from lightning.utils.configs import configurable


def read_bytes_images(paths, target_types=[]):
    img_path, target_paths = paths
    img = open(img_path, 'rb').read()
    targets = []
    for i, t in enumerate(target_types):
        targets.append(open(target_paths[i], 'rb').read())

    return img, tuple(targets)


class CityScapesWithDepth(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            mode: str = "fine",
            target_type: Union[List[str], str] = "instance",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            augments: Optional[List[Transform or Augmentation]] = None,
            target_augments: Optional[List[Transform or Augmentation]] = [],
            is_preload: bool = False
    ) -> None:
        super(CityScapesWithDepth, self).__init__(root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        print(self.images_dir)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        print(self.targets_dir)
        self.disparity_dir = os.path.join(self.root, 'disparity', split)
        print(self.disparity_dir)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.augments = augments
        self.target_augments = target_augments
        self.is_preload = is_preload

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semseg", "polygon", "color", "depth"))
         for value in self.target_type]

        # Check the data is usable
        if 'disparity' in self.target_type and not os.path.isdir(self.disparity_dir):
            print("disparity dir is not a directory in {}".format(self.disparity_dir))
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        dirs = os.listdir(self.images_dir)
        for city in dirs:
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            disparity_dir = os.path.join(self.disparity_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    if t == 'depth':
                        url = os.path.join(disparity_dir, target_name)
                    else:
                        url = os.path.join(target_dir, target_name)
                    target_types.append(url)

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

        if self.is_preload:
            self._preload()

    def _preload(self):
        from functools import partial
        imreader = partial(read_bytes_images, target_types=self.target_type)
        self.data_cache = preloader(self, imreader, zip(self.images, self.targets))

    def close(self):
        del self.data_cache

    @staticmethod
    def decode_bytes_images(bytes_images, use_depth=False, use_inverse_depth=False, target_type=[]):
        from io import BytesIO
        b_img, b_targets = bytes_images

        img = read_image(BytesIO(b_img), format='RGB')
        targets = []
        for i, t in enumerate(target_type):
            t_l = t.lower()
            if t_l == 'depth':
                depth_array = np.asarray(bytearray(b_targets[i]), dtype='uint8')
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED).astype(np.float32)  # unchanged is -1
                depth[depth > 0.0] = (depth[depth > 0.0] - 1.) / 256.0  # output disparity
                if use_depth:
                    depth[depth > 0.0] = (0.209313 * 2262.52) / depth[depth > 0.0]
                if use_inverse_depth:
                    depth[depth > 0.0] = depth[depth > 0.0] / (0.209313 * 2262.52)
                targets.append(depth)
            elif t_l == 'semseg':
                target = read_image(BytesIO(b_targets[i]), format='L')
                target = target.squeeze(2)
                targets.append(target)
            else:
                raise NotImplementedError("Not Support Type like {}.".format(t))
        return img, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        img, targets = self._read_img(index)

        img, transforms = T.apply_transform_gens(self.augments, img)
        output_transforms = self.target_augments[0].get_transform(img) if len(self.target_augments) > 0 else None

        post_targets = []
        for i, t in enumerate(self.target_type):
            post_target = self.post_process_target(targets[i], transforms, output_transforms, t)
            post_targets.append(post_target)

        img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)), dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(img)

        return img, tuple(post_targets)

    def _read_img(self, index: int) -> Tuple[Any, Any]:
        if self.is_preload:
            assert hasattr(self, 'data_cache') and self.data_cache is not None
            return CityScapesWithDepth.decode_bytes_images(self.data_cache[index], target_type=self.target_type)
        img = read_image(self.images[index], format='RGB').copy()
        targets = []
        instance_id, semseg_id = None, None
        for i, t in enumerate(self.target_type):
            if t == 'depth':
                # plt.imread return an numpy array, follow mtan data.
                # target = plt.imread(self.targets[index][i])
                # follow official cityscapes
                target = cv2.imread(self.targets[index][i], cv2.IMREAD_UNCHANGED).astype(np.float32)  # unchanged is -1
                target[target > 0.0] = (target[target>0.0] - 1.) / 256.0  # output disparity
                # target[target > 0.0] = (0.209313 * 2262.52) / target[target > 0.0]
            elif t == 'semseg':
                semseg_id = i
                format = 'L'
                target = read_image(self.targets[index][i], format=format)
                target = target.squeeze(2)
            elif t == 'instance':
                instance_id = i
                target = cv2.imread(self.targets[index][i], cv2.IMREAD_UNCHANGED)
            else:
                raise NotImplementedError("Not Support type like {}".format(t))
            targets.append(target.copy())
        if instance_id is not None:
            if semseg_id is not None:
                mask = targets[semseg_id]
            else:
                format = 'L'
                semseg_name = '_'.join(self.targets[index][instance_id].split('_')[:-1]) + '_labelTrainIds.png'
                target = read_image(semseg_name, format=format)
                mask = target.squeeze(2)
            y_map, x_map = encoding_instancemap(mask, targets[instance_id])
            targets[instance_id] = np.stack([y_map, x_map], axis=2)

        return img, targets

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semseg':
            return '{}_labelTrainIds.png'.format(mode)  # labelIds is original label(34 class), labelTrainIds is 19 classes label.
        elif target_type == 'depth':
            return 'disparity.png'
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

    def post_process_target(self, target, transforms, output_transforms, target_type):
        if target_type == 'depth':
            depth = transforms.apply_image(target).copy()
            if output_transforms is not None:
                depth = output_transforms.apply_image(depth)
            return torch.as_tensor(np.ascontiguousarray(depth))
        elif target_type == 'semseg':
            label = transforms.apply_segmentation(target).copy()
            if output_transforms is not None:
                label = output_transforms.apply_segmentation(label)
            return torch.as_tensor(label.astype("long"), dtype=torch.int64)
        elif target_type == 'instance':
            ins = transforms.apply_image(target).copy()
            if output_transforms is not None:
                ins = output_transforms.apply_image(ins)
            ins = ins.transpose(2, 0, 1)
            return torch.as_tensor(np.ascontiguousarray(ins))
        else:
            raise NotImplementedError("NYUv2 Not Support target type: {}".format(target_type))


@DATA_REGISTRY.register()
class CityscapesDataModule(pl.LightningDataModule):

    @configurable
    def __init__(
        self, 
        on_server,
        data_dir,
        target_type,
        in_resolution,
        out_resolution,
        is_preload,
        pixel_mean,
        pixel_std,
        eval_cfg,
        dataloader_cfg,
        gpus
    ):
        super().__init__()
        self.on_server = on_server
        self.data_dir = data_dir
        self.target_type = target_type
        self.in_resolution = in_resolution
        self.out_resolution = out_resolution
        self.dataloader_cfg = dataloader_cfg
        self.eval_cfg = eval_cfg
        self.gpus = gpus
        self.augments = [
            T.RandomCrop("relative_range", (0.5, 0.5)),
            T.Resize(in_resolution),
            T.RandomFlip()
        ]
        self.target_augments = [T.Resize(out_resolution)] if in_resolution != out_resolution else []
        # Test的时候还要做Augment就挺奇怪的, 应该只有一个Resize才对。
        self.test_augments = [
            # T.RandomCrop("relative_range", (0.5, 0.5)),
            T.Resize(in_resolution),
        ]
        self.pixel_mean, self.pixel_std = pixel_mean, pixel_std
        self.normalize_t = transforms.Normalize(mean=pixel_mean, std=pixel_std)
        self.is_preload = is_preload

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, self.in_resolution[0], self.in_resolution[1])
    
    @classmethod
    def from_config(cls, cfg):
        on_server = cfg.dataset.dataloader.on_server
        if on_server:
            data_dir = "/tmp_data/cityscapes"
        else:
            data_dir = os.environ.get("DETECTRON2_DATASETS")
            if data_dir is None:
                try:
                    data_dir = os.path.join(hydra.utils.get_original_cwd(), "data", "cityscapes")
                except:
                    data_dir = os.path.join('./data', "cityscapes")
        is_preload = ("is_preload" in cfg.dataset.dataloader) and cfg.dataset.dataloader.is_preload
        return {
            "on_server": cfg.dataset.dataloader.on_server,
            "data_dir": data_dir,
            "target_type":  [t.lower() for t in cfg.dataset.task.tasks], 
            "in_resolution": cfg.dataset.meta_data.in_resolution,
            "out_resolution": cfg.dataset.meta_data.out_resolution,
            "is_preload": is_preload,
            "pixel_mean": cfg.dataset.meta_data.pixel_mean,
            "pixel_std": cfg.dataset.meta_data.pixel_std,
            "dataloader_cfg": dict(cfg.dataset.dataloader),
            "eval_cfg": dict(cfg.dataset.eval),
            "gpus": cfg.dataset.gpus
        }

    def setup(self, stage: Optional[str] = None):

        self.train_dataset = self.dataset_type(
            self.data_dir,
            split='train',
            augments=self.augments,
            target_augments=self.target_augments,
            target_type=self.target_type,
            transform=self.normalize_t,
            is_preload=self.is_preload
        )
        self.val_dataset = self.dataset_type(
            self.data_dir,
            split='val',
            augments=self.test_augments,
            target_augments=self.target_augments,
            target_type=self.target_type,
            transform=self.normalize_t,
            is_preload=self.is_preload
        )


        # if stage == 'test' or stage is None:
        #     self.nyuv2_test = NYUv2(self.data_dir, split='test', transform=self.test_transform, out_transform=self.out_transform)

    @property
    def dataset_type(self):
        # if self.on_server:
        #     return CityScapesHDF5
        return CityScapesWithDepth

    def train_dataloader(self):
        batch_size = self.dataloader_cfg["batch_size"] // self.gpus
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            # num_workers=0,
            num_workers=self.dataloader_cfg["num_workers"],
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_cfg["pin_memory"] if 'pin_memory' in self.dataloader_cfg else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_cfg["batch_size"],
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_cfg["batch_size"], num_workers=4)

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()


@DATA_REGISTRY.register()
class CityscapesSADataModule(CityscapesDataModule):
    """
    This DataModule use one dataloader to sample SA-batch.
    """

    def train_dataloader(self):
        from lightning.data.sampler import SASampler
        sampler = SASampler(
            data_source=self.train_dataset,
            batch_size=self.cfg.train_setting.batch_size,
            drop_last=True,
            num_tasks=len(self.target_type)
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.cfg.train_setting.num_workers
        )


@DATA_REGISTRY.register()
class CityscapesSAInterpDataModule(CityscapesDataModule):
    """
    This DataModule use one dataloader to sample SA-batch.
    """

    @configurable
    def __init__(
        self, 
        *,
        sa_ratio,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sa_ratio = sa_ratio
    
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['sa_ratio'] = cfg.dataset.solver.sa_ratio
        return ret

    def train_dataloader(self):
        from lightning.data.sampler import SAInterpSampler
        sampler = SAInterpSampler(
            data_source=self.train_dataset,
            batch_size=self.dataloader_cfg["batch_size"],
            drop_last=True,
            num_tasks=len(self.target_type),
            sa_ratio=self.sa_ratio
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.dataloader_cfg["num_workers"]
        )


@DATA_REGISTRY.register()
class CityscapesSAITAPlusDataModule(CityscapesDataModule):
    """
    This DataModule use one dataloader to sample SA-ITA batch with fix number inserted.
    """

    @configurable
    def __init__(
            self,
            *,
            group_size,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.group_size = group_size

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['group_size'] = cfg.dataset.dataloader.group_size
        return ret

    def train_dataloader(self):
        from lightning.data.sampler import SAITAPlusSampler
        sampler = SAITAPlusSampler(
            data_source=self.train_dataset,
            batch_size=self.dataloader_cfg["batch_size"],
            drop_last=True,
            num_tasks=len(self.target_type),
            group_size=self.group_size
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.dataloader_cfg["num_workers"]
        )


@DATA_REGISTRY.register()
class CityscapesMTMODataModule(pl.LightningDataModule):

    @configurable
    def __init__(
        self, 
        on_server,
        data_dir,
        target_type,
        in_resolution,
        out_resolution,
        is_preload,
        eval_cfg,
        dataloader_cfg,
        gpus
    ):
        super().__init__()
        self.on_server = on_server
        self.data_dir = data_dir
        self.target_type = target_type
        self.in_resolution = in_resolution
        self.out_resolution = out_resolution
        self.dataloader_cfg = dataloader_cfg
        self.eval_cfg = eval_cfg
        self.gpus = gpus
        self.is_preload = is_preload

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, self.in_resolution[0], self.in_resolution[1])
    
    @classmethod
    def from_config(cls, cfg):
        on_server = cfg.dataset.dataloader.on_server
        if on_server:
            data_dir = "/tmp_data/cityscapes"
        else:
            data_dir = os.environ.get("DETECTRON2_DATASETS")
            if data_dir is None:
                try:
                    data_dir = os.path.join(hydra.utils.get_original_cwd(), "data", "cityscapes")
                except:
                    data_dir = os.path.join('./data', "cityscapes")
        is_preload = ("is_preload" in cfg.dataset.dataloader) and cfg.dataset.dataloader.is_preload
        return {
            "on_server": cfg.dataset.dataloader.on_server,
            "data_dir": data_dir,
            "target_type":  [t.lower() for t in cfg.dataset.task.tasks], 
            "in_resolution": cfg.dataset.meta_data.in_resolution,
            "out_resolution": cfg.dataset.meta_data.out_resolution,
            "is_preload": is_preload,
            "dataloader_cfg": dict(cfg.dataset.dataloader),
            "eval_cfg": dict(cfg.dataset.eval),
            "gpus": cfg.dataset.gpus
        }

    # def __init__(self, cfg):
    #     super().__init__()
    #     self.cfg = cfg
    #     self.on_server = self.cfg.train_setting.on_server
    #     self.data_dir = os.path.join(hydra.utils.get_original_cwd(), "data", "cityscapes") if not self.on_server else "/tmp_data/cityscapes"
    #     self.target_type = [t.lower() for t in self.cfg.task.tasks]
    #     self.is_preload = self.cfg.train_setting.is_preload if 'is_preload' in self.cfg.train_setting else False

    #     # self.dims is returned when you call dm.size()
    #     # Setting default dims here because we know them.
    #     # Could optionally be assigned dynamically in dm.setup()
    #     self.dims = (3, self.cfg.dataset.in_resolution[0], self.cfg.dataset.in_resolution[1])

    def setup(self, stage: Optional[str] = None):
        from lightning.data.mtmoloaders.cityscapes_loader import cityscapes_augmentations
        in_resolution = self.in_resolution
        img_size = (in_resolution[0], in_resolution[1])
        self.train_dataset = self.dataset_type(self.data_dir, split=['train'], is_transform=True, target_type=self.target_type, img_size=img_size, is_preload=self.is_preload, augmentations=cityscapes_augmentations)
        self.val_dataset = self.dataset_type(self.data_dir, split=['val'], is_transform=True, target_type=self.target_type, img_size=img_size, is_preload=self.is_preload)

    @property
    def dataset_type(self):
        from lightning.data.mtmoloaders.cityscapes_loader import CITYSCAPESWrapper
        return CITYSCAPESWrapper

    def train_dataloader(self):
        batch_size = self.dataloader_cfg["batch_size"] // self.gpus
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            # num_workers=0,
            num_workers=self.dataloader_cfg["num_workers"],
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_cfg["pin_memory"] if 'pin_memory' in self.dataloader_cfg else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_cfg["batch_size"],
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_cfg["batch_size"], num_workers=4)

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()


class CityscapesMTMOSADataModule(CityscapesMTMODataModule):

    def train_dataloader(self):
        from lightning.data.sampler import SAInterpSampler
        sampler = SAInterpSampler(
            data_source=self.train_dataset,
            batch_size=self.cfg.dataset.dataloader.batch_size,
            drop_last=True,
            num_tasks=len(self.target_type),
            sa_ratio=self.cfg.dataset.solver.sa_ratio
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.cfg.dataset.dataloader.num_workers,
            pin_memory = self.cfg.dataset.dataloader.pin_memory if 'pin_memory' in self.cfg.train_setting else False,
        )


# class CityscapesNPYDataModule(CityscapesDataModule):
#
#     def __init__(self, cfg):
#         super(CityscapesNPYDataModule, self).__init__(cfg)
#         self.data_dir = os.path.join(hydra.utils.get_original_cwd(), "data", "cityscapes_npy_small")
#
#     @property
#     def dataset_type(self):
#         return CityScapesNPY
#
#     def setup(self, stage: Optional[str] = None):
#
#         # Assign train/val dataset for use in dataloaders
#
#         params = {
#             'augments': self.augments,
#             'target_augments': self.target_augments,
#             'target_type': self.target_type,
#             'is_ram': self.is_ram,
#             # 'mode': self.cfg.dataset.semseg.num_classes
#             'mode': 7
#         }
#
#         if stage == 'fit' or stage is None:
#             self.train_dataset = self.dataset_type(self.data_dir, split='train', **params)
#             self.val_dataset = self.dataset_type(self.data_dir, split='val', **params)
#         elif stage == 'val':
#             self.val_dataset = self.dataset_type(self.data_dir, split='val', **params)


def show_img(batch, num_classes=19):
    from multitask.data.NYUv2 import NYUv2Dataset
    img, targets = batch
    semseg, depth = targets
    print(img.max(), img.min())
    print(semseg[semseg != 255].max(), semseg[semseg != 255].min())
    print(depth.max(), depth.min())
    print(depth.dtype)
    print(img.shape, semseg.shape, depth.shape)
    # tmp = (img[0]*255).numpy().astype(np.uint8)
    # print(tmp.max(), tmp.min())
    def convert2numpy(obj):
        if isinstance(obj, torch.Tensor):
            return obj.numpy()
        return obj
    pixel_mean = torch.tensor([103.53, 116.28, 123.675]).reshape(3, 1, 1)
    pixel_std = torch.tensor([57.375, 57.12, 58.395]).reshape(3, 1, 1)
    img = img * pixel_std + pixel_mean
    img, depth, semseg = convert2numpy(img), convert2numpy(depth), convert2numpy(semseg)
    data = {
        'img': img.astype(np.uint8).transpose(1, 2, 0),
        'depth': depth,
        'sem_seg': semseg
    }
    # NYUv2Dataset.data_show(data, num_classes=35)
    NYUv2Dataset.data_show(data, num_classes=num_classes)


def save_cityscapes_hdf5(split='train'):
    dataset = CityScapesWithDepth(
        root="/home/lyz/PycharmProjects/multitask/data/cityscapes",
        split=split,
        target_type=['semseg', 'depth']
    )
    # show_img(dataset[0])
    length = dataset.__len__()
    import h5py
    from tqdm import tqdm
    with h5py.File("/home/lyz/PycharmProjects/multitask/data/cityscapes_{}.hdf5".format(split), 'w') as f:
        imgs = f.create_dataset("images", shape=(length, 1024, 2048, 3), dtype='uint8')
        semseg = f.create_dataset("semseg", shape=(length, 1024, 2048))
        depth = f.create_dataset("depth", shape=(length, 1024, 2048))
        for i in tqdm(range(dataset.__len__())):
            img, targets = dataset._read_img(i)
            imgs[i] = img.astype(np.uint8)
            semseg[i] = targets[0]
            depth[i] = targets[1]


Void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1] + [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
def encoding_instancemap(mask: np.ndarray, ins: np.ndarray, ignore_idx: int = 255, no_instances: List[int]=Void_classes):
    """
    Adapt from github.com/intel-isl/MultiObjectiveOptimization/blob/master/multi_task/loaders/cityscapes_loader.py
    """
    ins[mask == ignore_idx] = ignore_idx
    for no_ins in no_instances:
        ins[ins == no_ins] = ignore_idx
    ins[ins == 0] = ignore_idx

    instances_ids = np.unique(ins)
    ins_size = ins.shape
    # print(ins_size)
    ymap, xmap = np.meshgrid(np.arange(ins_size[0]), np.arange(ins_size[1]), indexing='ij')

    # out_ymap, out_xmap = np.ones_like(ymap) * ignore_idx, np.ones_like(xmap) * ignore_idx
    out_ymap, out_xmap = np.meshgrid(np.arange(ins_size[0]), np.arange(ins_size[1]), indexing='ij')
    out_ymap = np.ones(ymap.shape) * ignore_idx
    out_xmap = np.ones(xmap.shape) * ignore_idx

    for id in instances_ids:
        if id == ignore_idx:
            continue
        instance_indicator = (ins == id)
        coordinate_y, coordinate_x = np.mean(ymap[instance_indicator]), np.mean(xmap[instance_indicator])
        out_ymap[instance_indicator] = ymap[instance_indicator] - coordinate_y
        out_xmap[instance_indicator] = xmap[instance_indicator] - coordinate_x

    # print(out_ymap.min(), out_ymap.max())
    # print(out_xmap.min(), out_xmap.max())

    return out_ymap.astype(np.float32), out_xmap.astype(np.float32)


if __name__ == "__main__":
    split = 'train'
    # save dataset to hdf5
    # save_cityscapes_hdf5(split=split)

    # Test Dataset
    dataset = CityScapesWithDepth(
        root="/home/lyz/PycharmProjects/multitask/data/cityscapes",
        split=split,
        target_type=['semseg', 'depth'],
        augments=[T.Resize((128, 256))],
        transform=transforms.Normalize(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]),
        # is_preload=True
    )
    batch = dataset[1000]
    show_img(batch)
    img, targets = batch

    # To show Instances
    # plt.imshow(targets[2][0, :, :])
    # plt.show()
    # plt.imshow(targets[2][1, :, :])
    # plt.show()
    # show_img(batch)
    # dataset = CityScapesHDF5(
    #     root="/home/lyz/PycharmProjects/multitask/data/",
    #     split=split,
    #     target_type=['semseg', 'depth'],
    #     augments=[T.Resize((128, 256))]
    # )
    # import time
    # import random
    # start_t = time.time()
    # for i in range(100):
    #     dataset[i+random.randint(1, 300)]
    # end_t = time.time()
    # print((end_t - start_t) / 100.0)

