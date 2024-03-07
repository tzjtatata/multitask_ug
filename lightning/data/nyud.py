# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import tarfile
import cv2

from PIL import Image
from typing import Optional
import numpy as np
import torch.utils.data as data
import pytorch_lightning as pl
from torchvision import transforms

from lightning.data.mypath import MyPath
from lightning.utils.utils import mkdir_if_missing
from lightning.data.google_drive import download_file_from_google_drive
from lightning.utils.configs import configurable
from torch.utils.data import DataLoader
from .regist_dataset import DATA_REGISTRY
from .custom_collate import collate_mil, collate_mil_for_MuST


FLAGVALS = {
    'image': cv2.INTER_CUBIC,
    'semseg': cv2.INTER_NEAREST,
    'depth': cv2.INTER_NEAREST,
    'normals': cv2.INTER_CUBIC
}

INFER_FLAGVALS = {
    'semseg': cv2.INTER_NEAREST,
    'depth': cv2.INTER_LINEAR,
    'normals': cv2.INTER_LINEAR
}

# NYUD官方给出的40类表如下：https://blog.csdn.net/weixin_43915709/article/details/88774325


"""
    Transformations, datasets and dataloaders
"""
# 这里要特别注意！ 这里的transform是根据传入的sample dict的key来决定执行操作的。
# 如果传入的Key不在以上的INFER_FLAGVALS和FLAGVALS里的话，那么样本会被删除！！
def get_transformations(db_name, resolution):
    """ Return transformations for training and evaluationg """
    from lightning.data import custom_transforms as tr

    # Training transformations
    # transforms_tr = []
    if db_name.startswith('NYU'):
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]

        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=[0], scales=[1.0, 1.2, 1.5],
                                              flagvals={x: FLAGVALS[x] for x in FLAGVALS})])
        pass

    elif db_name == 'PASCALContext':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]

        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                              flagvals={x: FLAGVALS[x] for x in FLAGVALS})])

    else:
        raise ValueError('Invalid train db name'.format(db_name))

    # Fixed Resize to input resolution
    # transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(resolution) for x in FLAGVALS},
                                        #  flagvals={x: FLAGVALS[x] for x in FLAGVALS})])

    # 这是正确的版本，暂时为了Detection测试把Normalize删掉了。
    transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor()])
    transforms_tr = transforms.Compose(transforms_tr)

    # Testing (during training transforms)
    transforms_ts = []
    transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(resolution) for x in FLAGVALS},
                                         flagvals={x: FLAGVALS[x] for x in FLAGVALS})])

    # 这是正确的版本，暂时为了Detection测试把Normalize删掉了。
    transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor()])
    transforms_ts = transforms.Compose(transforms_ts)

    return transforms_tr, transforms_ts


def get_train_dataset(db_name, transforms, tasks):
    """ Return the train dataset """
    print('Preparing train loader for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(split=['train'], transform=transforms, retname=True,
                                 do_semseg='semseg' in p.ALL_TASKS.NAMES,
                                 do_edge='edge' in p.ALL_TASKS.NAMES,
                                 do_normals='normals' in p.ALL_TASKS.NAMES,
                                 do_sal='sal' in p.ALL_TASKS.NAMES,
                                 do_human_parts='human_parts' in p.ALL_TASKS.NAMES,
                                 overfit=p['overfit'])

    elif db_name == 'NYUMT':
        print(transforms)
        database = NYU_MT(split='train', transform=transforms, do_edge='edge' in tasks,
                           do_semseg='semseg' in tasks,
                           do_normals='normals' in tasks,
                           do_depth='depth' in tasks, overfit=False)

    elif db_name == 'NYUD':
        from lightning.data.nyu_full import NYUD
        database = NYUD(split='train', transform=transforms, overfit=False)
    
    elif db_name == 'NYUFull':
        from lightning.data.nyu_full import NYUFull
        database = NYUFull(split='train', transform=transforms, overfit=False, max_samples=2000)

    else:
        raise NotImplemented("train_db_name: Choose among PASCALContext and NYUD")

    return database


def get_train_dataloader(dataset, trBatch, nworkers, is_must=False):
    """ Return the train dataloader """
    if is_must:
        collate_fn = collate_mil_for_MuST
    else:
        collate_fn = collate_mil
    trainloader = DataLoader(dataset, batch_size=trBatch, shuffle=True, drop_last=True,
                             num_workers=nworkers, collate_fn=collate_fn, persistent_workers=True)
    return trainloader


def get_val_dataset(db_name, transforms, tasks):
    """ Return the validation dataset """

    print('Preparing val loader for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(split=['val'], transform=transforms, retname=True,
                                 do_semseg='semseg' in p.TASKS.NAMES,
                                 do_edge='edge' in p.TASKS.NAMES,
                                 do_normals='normals' in p.TASKS.NAMES,
                                 do_sal='sal' in p.TASKS.NAMES,
                                 do_human_parts='human_parts' in p.TASKS.NAMES,
                                 overfit=p['overfit'])

    elif db_name == 'NYUMT':
        database = NYU_MT(split='val', transform=transforms, do_edge='edge' in tasks,
                           do_semseg='semseg' in tasks,
                           do_normals='normals' in tasks,
                           do_depth='depth' in tasks, overfit=False)
    
    elif db_name == 'NYUD':
        from lightning.data.nyu_full import NYUD
        database = NYUD(split='val', transform=transforms, overfit=False)
    
    elif db_name == 'NYUFull':
        from lightning.data.nyu_full import NYUFull
        database = NYUFull(split='val', transform=transforms, overfit=False)

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_val_dataloader(dataset, valBatch, nworkers):
    """ Return the validation dataloader """
    testloader = DataLoader(dataset, batch_size=valBatch, shuffle=False, drop_last=False,
                            num_workers=nworkers, persistent_workers=True)
    return testloader


class NYU_MT(data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes semantic segmentation and depth prediction.

    Data can also be found at:
    https://drive.google.com/file/d/14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw/view?usp=sharing

    """

    GOOGLE_DRIVE_ID = '14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw'
    FILE = 'NYUD_MT.tgz'

    def __init__(self,
                 root=MyPath.db_root_dir('NYUD_MT'),
                 download=True,
                 split='val',
                 transform=None,
                 retname=True,
                 overfit=False,
                 do_edge=False,
                 do_semseg=False,
                 do_normals=False,
                 do_depth=False,
                 ):

        self.root = root

        if download:
            self._download()

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
        _image_dir = os.path.join(root, 'images')
        
        # Edge Detection
        self.do_edge = do_edge
        self.edges = []
        _edge_gt_dir = os.path.join(root, 'edge')

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []
        _semseg_gt_dir = os.path.join(root, 'segmentation')

        # Surface Normals
        self.do_normals = do_normals
        self.normals = []
        _normal_gt_dir = os.path.join(root, 'normals')

        # Depth
        self.do_depth = do_depth
        self.depths = []
        _depth_gt_dir = os.path.join(root, 'depth')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'gt_sets')

        print('Initializing dataloader for NYUD {} set'.format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), 'r') as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                # Images
                _image = os.path.join(_image_dir, line + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Edges
                _edge = os.path.join(self.root, _edge_gt_dir, line + '.npy')
                assert os.path.isfile(_edge)
                self.edges.append(_edge)

                # Semantic Segmentation
                _semseg = os.path.join(self.root, _semseg_gt_dir, line + '.png')
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                # Surface Normals
                _normal = os.path.join(self.root, _normal_gt_dir, line + '.npy')
                assert os.path.isfile(_normal)
                self.normals.append(_normal)

                # Depth Prediction
                _depth = os.path.join(self.root, _depth_gt_dir, line + '.npy')
                assert os.path.isfile(_depth)
                self.depths.append(_depth)

        if self.do_edge:
            assert (len(self.images) == len(self.edges))
        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_depth:
            assert (len(self.images) == len(self.depths))
        if self.do_normals:
            assert (len(self.images) == len(self.normals))

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

        if self.do_edge:
            _edge = self._load_edge(index)
            if _edge.shape != _img.shape[:2]:
                _edge = cv2.resize(_edge, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['edge'] = _edge

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape != _img.shape[:2]:
                print('RESHAPE SEMSEG')
                _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals(index)
            if _normals.shape[:2] != _img.shape[:2]:
                _normals = cv2.resize(_normals, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            if _depth.shape[:2] != _img.shape[:2]:
                print('RESHAPE DEPTH')
                _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_edge(self, index):
        _edge = np.load(self.edges[index]).astype(np.float32)
        return _edge

    def _load_semseg(self, index):
        # Note: We ignore the background class as other related works.
        _semseg = np.array(Image.open(self.semsegs[index])).astype(np.float32)
        _semseg[_semseg == 0] = 256
        _semseg = _semseg - 1
        return _semseg

    def _load_depth(self, index):
        _depth = np.load(self.depths[index])
        return _depth

    def _load_normals(self, index):
        _normals = np.load(self.normals[index])
        return _normals

    def _download(self):
        _fpath = os.path.join(MyPath.db_root_dir(), self.FILE)
        _dpath = _fpath[:-4]
        print(_dpath)

        if os.path.isfile(_fpath) or os.path.exists(_dpath):
            print('Files already downloaded')
        else:
            print('Downloading from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)
        
        if os.path.exists(_dpath) and os.path.isdir(_dpath):
            print("Data have been done.")
            return

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(MyPath.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def __str__(self):
        return 'NYUD Multitask (split=' + str(self.split) + ')'


@DATA_REGISTRY.register()
class NYUMTDataModule(pl.LightningDataModule):

    @configurable
    def __init__(
        self,
        db_name,
        tasks,
        resolution,
        trBatch,
        valBatch,   
        nworkers,
        is_transform_for_train=True
    ):
        super().__init__()
        self.db_name = db_name
        self.tasks = tasks
        self.resolution = resolution
        self.trBatch = trBatch
        self.valBatch = valBatch
        self.nworkers = nworkers

        self.train_transforms, self.val_transforms = get_transformations(self.db_name, self.resolution)
        self.dims = (3, self.resolution[0], self.resolution[1])
        self.is_transform_for_train = is_transform_for_train

    @classmethod
    def from_config(cls, cfg):
        tasks = cfg.dataset.task.tasks
        if 'aux_tasks' in cfg.dataset.task:
            tasks += cfg.dataset.task.aux_tasks
        return {
            'db_name': cfg.dataset.name,
            'tasks': tasks,
            'resolution': cfg.dataset.meta_data.in_resolution,
            'trBatch': cfg.dataset.dataloader.batch_size // cfg.dataset.gpus,
            'valBatch': cfg.dataset.eval.batch_size,
            'nworkers': cfg.dataset.dataloader.num_workers
        }
    
    def get_train_dataset_with_val_transform(self):
        return get_train_dataset(self.db_name, self.val_transforms, self.tasks)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val dataset for use in dataloaders
        if self.is_transform_for_train:
            self.train_dataset = get_train_dataset(self.db_name, self.train_transforms, self.tasks)
        else:
            self.train_dataset = get_train_dataset(self.db_name, self.val_transforms, self.tasks)
        self.val_dataset = get_val_dataset(self.db_name, self.val_transforms, self.tasks)
        # self.test_dataset = get_val_dataset(self.db_name, None, self.tasks)

    def train_dataloader(self):
        is_must = (self.db_name == 'NYUFull')
        return get_train_dataloader(self.train_dataset, self.trBatch, self.nworkers, is_must=is_must)

    def val_dataloader(self):
        return get_val_dataloader(self.val_dataset, self.valBatch, self.nworkers)

    def test_dataloader(self):
        return get_val_dataloader(self.val_dataset, self.valBatch, self.nworkers)


def test_mt():
    import torch
    import lightning.data.custom_transforms as tr
    import  matplotlib.pyplot as plt 
    from torchvision import transforms
    transform = transforms.Compose([tr.RandomHorizontalFlip(),
                                    tr.ScaleNRotate(rots=(-2, 2), scales=(.75, 1.25),
                                                    flagvals={'image': cv2.INTER_CUBIC,
                                                              'edge': cv2.INTER_NEAREST,
                                                              'semseg': cv2.INTER_NEAREST,
                                                              'normals': cv2.INTER_LINEAR,
                                                              'depth': cv2.INTER_LINEAR}),
                                    tr.FixedResize(resolutions={'image': (512, 512),
                                                                'edge': (512, 512),
                                                                'semseg': (512, 512),
                                                                'normals': (512, 512),
                                                                'depth': (512, 512)},
                                                   flagvals={'image': cv2.INTER_CUBIC,
                                                             'edge': cv2.INTER_NEAREST,
                                                             'semseg': cv2.INTER_NEAREST,
                                                             'normals': cv2.INTER_LINEAR,
                                                             'depth': cv2.INTER_LINEAR}),
                                    tr.AddIgnoreRegions(),
                                    tr.ToTensor()])
    dataset = NYU_MT(split='train', transform=transform, retname=True,
                      do_edge=True,
                      do_semseg=True,
                      do_normals=True,
                      do_depth=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=5)

    for i, sample in enumerate(dataloader):
        print(i)
        for j in range(sample['image'].shape[0]):
            f, ax_arr = plt.subplots(5)
            for k in range(len(ax_arr)):
                ax_arr[k].cla()
            ax_arr[0].imshow(np.transpose(sample['image'][j], (1,2,0)))
            ax_arr[1].imshow(sample['edge'][j,0])
            ax_arr[2].imshow(sample['semseg'][j,0]/40)
            ax_arr[3].imshow(np.transpose(sample['normals'][j], (1,2,0)))
            max_depth = torch.max(sample['depth'][j,0][sample['depth'][j,0] != 255]).item()
            ax_arr[4].imshow(sample['depth'][j,0]/max_depth) # Not ideal. Better is to show inverse depth.

            plt.show()
        break


if __name__ == '__main__':
    test_mt()
