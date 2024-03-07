import math
import os
import pytorch_lightning as pl
import torch
import h5py
from torch.utils.data import random_split, DataLoader
from lightning.data.utils import loop_until_success
import hydra
import numpy as np
import scipy.io as scio

from typing import (
    Optional,
    Callable,
    Union,
    List,
    Tuple,
    Any
)
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg, iterable_to_str
from lightning.data import transforms as TL
from lightning.data.transforms import Augmentation
from lightning.data.utils import read_image, convert_PIL_to_numpy
from lightning.utils.configs import configurable
from PIL import Image
from fvcore.transforms import Transform
from .regist_dataset import DATA_REGISTRY


target2postfix = {
    "semseg": "semseg.png",
    "depth": "depth.tiff"
}


class NYUv2(VisionDataset):
    """`NYUv2 <https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory of Label13
        split (string, optional): The image split to use, ``train``, ``test`` or ``val``
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

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "instance",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            augments: Optional[List[Transform or Augmentation]] = None,
            target_augments: Optional[List[Transform or Augmentation]] = None,
            use_data_interval: Optional[Tuple[int, int]] = None,
            is_shuffle: bool = False
    ) -> None:
        super(NYUv2, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(self.root, 'labels13', split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.augments = augments
        self.target_augments = target_augments

        valid_modes = ("train", "test", "val")
        msg = ("Unknown value '{}' for argument split. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("semseg", "depth"))
         for value in self.target_type]

        dirs = os.listdir(self.images_dir)

        for ids in dirs:
            if ids.endswith(".mat"):
                continue
            img_dir = os.path.join(self.images_dir, ids)
            target_types = []
            for t in self.target_type:
                target_name = '{}_{}'.format(ids, target2postfix[t])
                target_types.append(os.path.join(img_dir, target_name))

            self.images.append(os.path.join(img_dir, ids+'.png'))
            self.targets.append(target_types)
        
        # 把数据打包，以便后续的裁剪和随机
        zipped_data = list(zip(self.images, self.targets))
        # 把数据进行裁剪，-1意味着自适应取值。
        if use_data_interval is not None:
            start, end = use_data_interval
            start = 0 if start == -1 else start
            end = len(zipped_data) if end == -1 else end
            zipped_data = zipped_data[start:end]
        
        # 把数据顺序打乱
        if is_shuffle:
            np.random.shuffle(zipped_data)

        self.images, self.targets = zip(*zipped_data)

    @loop_until_success
    def _read_img(self, index: int) -> Tuple[Any, Any]:
        img = read_image(self.images[index], format='RGB').copy()
        targets = []
        for i, t in enumerate(self.target_type):
            format = 'F' if t=='depth' else 'L'
            target = read_image(self.targets[index][i], format=format)
            if format == 'L':
                target = target.squeeze(2)
            targets.append(target.copy())
        return img, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            a dict contains at most three keys: image, semseg, depth, according to target_type.
        """
        img, targets = self._read_img(index)

        # utils.check_image_size(data_dict, img)

        img, transforms = TL.apply_transform_gens(self.augments, img)
        output_transforms = self.target_augments[0].get_transform(img)

        post_targets = []
        for i, t in enumerate(self.target_type):
            post_target = self.post_process_target(targets[i], transforms, output_transforms, t)
            post_targets.append(post_target)

        img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)), dtype=torch.float32)

        return img, tuple(post_targets)

    def post_process_target(self, target, transforms, output_transforms, target_type):
        if target_type == 'depth':
            depth = transforms.apply_image(target).copy()
            depth = output_transforms.apply_image(depth)
            return torch.as_tensor(np.ascontiguousarray(depth))
        elif target_type == 'semseg':
            label = transforms.apply_segmentation(target).copy()
            label = output_transforms.apply_segmentation(label)
            return torch.as_tensor(label.astype("long"), dtype=torch.int64)
        else:
            raise NotImplementedError("NYUv2 Not Support target type: {}".format(target_type))

    def _open_target_image(self, image_path, target_type):
        if target_type == "semseg":
            return Image.open(image_path).convert("L")
        elif target_type == "depth":
            return Image.open(image_path).convert("F")

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)


class NYUv2Ram(VisionDataset):
    """`NYUv2 <https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory of Label13
        split (string, optional): The image split to use, ``train``, ``test`` or ``val``
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

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "instance",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            augments: Optional[List[Transform or Augmentation]] = None,
            target_augments: Optional[List[Transform or Augmentation]] = None,
            is_shuffle: bool = False
    ) -> None:
        super(NYUv2Ram, self).__init__(root, transforms, transform, target_transform)
        self.file = h5py.File(os.path.join(self.root, "nyu_depth_v2_labeled.mat"), "r")
        self.labels_file = scio.loadmat(os.path.join(self.root, "labels13_full.mat"))
        split_file = scio.loadmat(os.path.join(self.root, "splits.mat"))

        self.target_type = target_type
        self.images = []
        self.targets = []
        self.augments = augments
        self.target_augments = target_augments

        valid_modes = ("train", "test", "val")
        msg = ("Unknown value '{}' for argument split. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("semseg", "depth"))
         for value in self.target_type]

        split_idxs = split_file['trainNdxs'] if split == 'train' else split_file['testNdxs']
        if is_shuffle:
            np.random.shuffle(split_idxs)
        for ids in split_idxs:
            id = ids[0] - 1
            img = Image.fromarray(np.moveaxis(self.file['images'][id], 0, -1), mode='RGB').rotate(-90, expand=True)
            img = convert_PIL_to_numpy(img, 'RGB')

            target_types = []
            for t in self.target_type:
                target_types.append(self._read_img(id, t))

            self.images.append(img)
            self.targets.append(target_types)

    def _read_img(self, index: int, target_type: str) -> np.ndarray:
        if target_type == 'depth':
            depth = Image.fromarray(self.file['depths'][index], mode='F').rotate(-90, expand=True)
            depth = convert_PIL_to_numpy(depth, 'F')
            return depth
        elif target_type == 'semseg':
            label = Image.fromarray(self.labels_file['labels40'][index]).rotate(-90, expand=True)
            label = convert_PIL_to_numpy(label, 'L').squeeze(2)
            return label
        else:
            raise NotImplementedError("NYUv2 Not Support target type: {}".format(target_type))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            a dict contains at most three keys: image, semseg, depth, according to target_type.
        """
        img, targets = self.images[index], self.targets[index]
        img = img.copy()
        targets = [
            t.copy()
            for t in targets
        ]

        # utils.check_image_size(data_dict, img)

        img, transforms = TL.apply_transform_gens(self.augments, img)
        output_transforms = self.target_augments[0].get_transform(img)

        post_targets = []
        for i, t in enumerate(self.target_type):
            post_target = self.post_process_target(targets[i], transforms, output_transforms, t)
            post_targets.append(post_target)

        img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)), dtype=torch.float32)

        return img, tuple(post_targets)

    def post_process_target(self, target, transforms, output_transforms, target_type):
        if target_type == 'depth':
            depth = transforms.apply_image(target).copy()
            depth = output_transforms.apply_image(depth)
            return torch.as_tensor(np.ascontiguousarray(depth))
        elif target_type == 'semseg':
            label = transforms.apply_segmentation(target).copy()
            label = output_transforms.apply_segmentation(label)
            return torch.as_tensor(label.astype("long"), dtype=torch.int64)
        else:
            raise NotImplementedError("NYUv2 Not Support target type: {}".format(target_type))

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)


@DATA_REGISTRY.register()
class NYUv2DataModule(pl.LightningDataModule):

    @configurable
    def __init__(
        self, 
        data_dir,
        target_type,
        in_resolution,
        out_resolution,
        dataloader_cfg,
        eval_cfg,
        gpus
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target_type = target_type
        self.in_resolution = in_resolution
        self.out_resolution = out_resolution
        self.dataloader_cfg = dataloader_cfg
        self.eval_cfg = eval_cfg
        self.gpus = gpus

        self.augments = [
            TL.RandomCrop("relative_range", (0.5, 0.5)),
            TL.Resize(in_resolution),
            TL.RandomFlip()
        ]
        self.target_augments = [TL.Resize(out_resolution)]
        # Test的时候还要做Augment就挺奇怪的, 应该只有一个Resize才对。
        self.test_augments = [
            TL.Resize(in_resolution),
        ]

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, self.in_resolution[0], self.in_resolution[1])
    
    @classmethod
    def from_config(cls, cfg):
        on_server = cfg.dataset.dataloader.on_server
        if on_server:
            data_dir = "/tmp_data/NYUv2"
        else:
            data_dir = os.environ.get("DETECTRON2_DATASETS")
            if data_dir is None:
                try:
                    data_dir = os.path.join(hydra.utils.get_original_cwd(), "data", "NYUv2")
                except:
                    data_dir = os.path.join('./data', "NYUv2")
        return {
            "data_dir": data_dir,
            "target_type":  [t.lower() for t in cfg.dataset.task.tasks], 
            "in_resolution": cfg.dataset.meta_data.in_resolution,
            "out_resolution": cfg.dataset.meta_data.out_resolution,
            "dataloader_cfg": dict(cfg.dataset.dataloader),
            "eval_cfg": dict(cfg.dataset.eval),
            "gpus": cfg.dataset.gpus
        }
        
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val dataset for use in dataloaders
        self.train_dataset = self.dataset_type(self.data_dir, split='train', augments=self.augments, target_augments=self.target_augments, target_type=self.target_type)
        self.val_dataset = self.dataset_type(self.data_dir, split='test', augments=self.test_augments, target_augments=self.target_augments, target_type=self.target_type)

    @property
    def dataset_type(self):
        return NYUv2

    def train_dataloader(self):
        cfg_node = self.dataloader_cfg
        batch_size = cfg_node['batch_size']
        batch_size = batch_size // self.gpus
        print("Batch Size is {}.".format(batch_size))
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=cfg_node['num_workers'],
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_cfg['batch_size'],
            num_workers=self.dataloader_cfg['num_workers']
        )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_cfg['batch_size'], num_workers=self.dataloader_cfg['num_workers'])


@DATA_REGISTRY.register()
class NYUv2SAInterpDataModule(NYUv2DataModule):
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
            batch_size=self.dataloader_cfg['batch_size'],
            drop_last=True,
            num_tasks=len(self.target_type),
            sa_ratio=self.sa_ratio
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.dataloader_cfg['num_workers']
        )


@DATA_REGISTRY.register()
class NYUv2STAGapDataModule(NYUv2DataModule):
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
        from lightning.data.sampler import STAGapSampler
        sampler = STAGapSampler(
            data_source=self.train_dataset,
            batch_size=self.dataloader_cfg['batch_size'],
            drop_last=True,
            num_tasks=len(self.target_type),
            group_size=self.group_size
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.dataloader_cfg['num_workers']
        )


class NYUv2MAMultipleDataModule(NYUv2DataModule):

    def train_dataloader(self):
        from lightning.data.sampler import MultipleSampler
        sampler = MultipleSampler(
            data_source=self.train_dataset,
            batch_size=self.cfg.train_setting.batch_size,
            drop_last=True,
            num_tasks=len(self.target_type),
        )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.cfg.train_setting.num_workers
        )


# class NYUv2SAInterpDataModule(NYUv2DataModule):
#
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.is_split_data = self.cfg.train_setting.is_split_data
#         self.ratio = self.cfg.train_setting.sa_ratio
#
#     def setup(self, stage: Optional[str] = None):
#         split_point = int(795.0 * self.ratio)
#         self.train_dataset = {}
#         sadiag_augments = self.build_augments(mode="110")
#         self.train_dataset['SADiag'] = self.dataset_type(self.data_dir, split='train', augments=sadiag_augments[0],
#                                                         target_augments=sadiag_augments[1],
#                                                         target_type=self.target_type,
#                                                         use_data_interval=(-1, split_point) if self.is_split_data else None)
#         sa_datasets = {}
#         for i, t in enumerate(self.target_type):
#             augments = self.build_augments(mode="110")
#             sa_datasets[t] = self.dataset_type(self.data_dir, split='train', augments=augments[0],
#                                                 target_augments=augments[1],
#                                                 target_type=t,
#                                                 use_data_interval=(split_point, -1) if self.is_split_data else None)
#         self.train_dataset['SA'] = sa_datasets
#         self.val_dataset = self.dataset_type(self.data_dir, split='test', augments=self.test_augments,
#                                       target_augments=self.target_augments, target_type=self.target_type)
#
#     def train_dataloader(self):
#         origin_batchsize = self.cfg.train_setting.batch_size
#         sadiag_batchsize = math.ceil(origin_batchsize * self.ratio)
#         sa_batchsize = (origin_batchsize - sadiag_batchsize) // len(self.target_type)
#         loaders = {
#             'SADiag': DataLoader(
#                 self.train_dataset['SADiag'],
#                 batch_size=sadiag_batchsize,
#                 num_workers=4,
#                 shuffle=True,
#                 drop_last=True
#             ),
#             'SA': {
#                 t: DataLoader(
#                     self.train_dataset['SA'][t],
#                     batch_size=sa_batchsize,
#                     num_workers=2,
#                     shuffle=True,
#                     drop_last=True
#                 )
#                 for t in self.target_type
#             }
#         }
#         return loaders


@DATA_REGISTRY.register()
class NYUv2EvalDataModule(NYUv2DataModule):

    def setup(self, stage: Optional[str] = None):
        self.val_dataset = self.dataset_type(self.data_dir, split='train', augments=self.test_augments,
                                             target_augments=self.target_augments, target_type=self.target_type)

        # self.val_dataset = self.dataset_type(self.data_dir, split='train', augments=self.augments,
        #                                    target_augments=self.target_augments, target_type=self.target_type)


def handle_MS_batch(batch):
    from matplotlib import pyplot as plt
    from .NYUv2_utils import NYUv2_data_show

    rows = len(batch)
    for k, v in batch.items():
        fig = plt.figure("Labeled Dataset Sample: {}".format(k), figsize=(12, 5))
        counts = 1
        img, targets = v
        # data = {
        #     'img': img,
        #     k: targets[0]
        # }
        batch_size = img.shape[0]
        for i in range(batch_size):
            ax = fig.add_subplot(batch_size, 2, counts)
            image = img[i] if img[i].device == 'cpu' else img[i].cpu()
            NYUv2_data_show(ax, 'img', image.numpy().astype(np.uint8).transpose(1, 2, 0))
            ax = fig.add_subplot(batch_size, 2, counts+1)
            target = targets[0][i] if targets[0][i].device == 'cpu' else targets[0][i].cpu()
            print(target.shape)
            NYUv2_data_show(ax, k, target.numpy())
            counts += 2
        plt.show()


def handle_batch(batch, target_type=['semseg', 'depth']):
    from matplotlib import pyplot as plt
    from .NYUv2_utils import NYUv2_data_show

    img, targets = batch
    batch_size = img.shape[0]
    counts = 0
    rows = len(target_type) + 1
    fig = plt.figure("Labeled Dataset Sample", figsize=(12, 5))
    for i in range(batch_size):
        # data = {
        #     'img': img,
        #     k: targets[0]
        # }
        counts += 1
        ax = fig.add_subplot(batch_size, rows, counts)
        image = img[i] if img[i].device == 'cpu' else img[i].cpu()
        NYUv2_data_show(ax, 'img', image.numpy().astype(np.uint8).transpose(1, 2, 0))
        for j, k in enumerate(target_type):
            counts += 1
            ax = fig.add_subplot(batch_size, rows, counts)
            target = targets[j][i] if targets[j][i].device == 'cpu' else targets[j][i].cpu()
            if k == 'semseg' and len(targets[j][i].shape) == 3:
                target = torch.argmax(target, dim=0)
            NYUv2_data_show(ax, k, target.squeeze().numpy())
    plt.show()


def handle_new_batch(batch, target_type=['semseg', 'depth']):
    """
    Show batch which Format by SAUnified Module.
    """
    from matplotlib import pyplot as plt
    from .NYUv2_utils import NYUv2_data_show

    batch_type = len(batch)
    if batch_type == 2:
        imgs, targets = batch
        preds = None
    else:
        imgs, targets, preds = batch

    for i in range(len(imgs)):
        fig = plt.figure("Labeled Dataset Sample: {}".format(target_type[i]), figsize=(12, 5))
        counts = 1
        img, t = imgs[i], targets[i]
        batch_size = img.shape[0]
        for j in range(batch_size):
            ax = fig.add_subplot(batch_size, batch_type, counts)
            image = img[j] if img[j].device == 'cpu' else img[j].cpu()
            NYUv2_data_show(ax, 'img', image.numpy().astype(np.uint8).transpose(1, 2, 0))
            ax = fig.add_subplot(batch_size, batch_type, counts + 1)
            target = t[j] if t[j].device == 'cpu' else t[j].cpu()
            NYUv2_data_show(ax, target_type[i], target.numpy())
            if batch_type == 3:
                ax = fig.add_subplot(batch_size, batch_type, counts + 2)
                pred = preds[i][j] if preds[i][j].device == 'cpu' else preds[i][j].cpu()
                NYUv2_data_show(ax, target_type[i], pred.numpy())
            counts += batch_type
        plt.show()
