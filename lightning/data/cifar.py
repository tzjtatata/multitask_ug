import numpy as np
import os
import pytorch_lightning as pl
from typing import Optional, Callable
from torch.utils.data import DataLoader
import hydra
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms
from lightning.utils.configs import configurable
from lightning.data.regist_dataset import DATA_REGISTRY


class MultiTaskCIFAR10(CIFAR10):

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            ratio: float = 1.0,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        # @lyz: @myt we do nothing when ratio is 1.0
        self.ratio = ratio
        assert ratio >= 0.0 and ratio <=1.0, "ratio is illegal, when its value is {}".format(ratio)
        if ratio < 1.0:
            # @lyz: @myt hint message should be clear.
            print("Modifying CIFAR10 to {}% Data.".format(ratio*100))
            # @myt d = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            # @lyz
            d = {i: [] for i in range(10)}
            all = []
            new_target = []
            for i, target in enumerate(self.targets):
                d[target].append(self.data[i])
            for target in d.keys():
                np.random.shuffle(d[target])  # 随机化每个list图片
                # d[target] -> List[array[32, 32, 3]]
                # 取每个list前ratio*size的图片训练
                d[target] = d[target][:int(ratio * len(d[target]))]
                all = all+d[target]
                new_target += [target] * len(d[target])
            all = np.stack(all)
            self.data = all
            self.targets = new_target
            print("Finished Modifying, Now Dataset has {} images".format(len(self.data)))

    def __getitem__(self, item):
        img, targets = super().__getitem__(item)
        return img, [targets, ]


class MultiTaskCIFAR100(CIFAR100):

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            ratio: float = 1.0,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.ratio = ratio
        assert ratio >= 0.0 and ratio <=1.0, "ratio is illegal, when its value is {}".format(ratio)

    def __getitem__(self, item):
        img, targets = super().__getitem__(item)
        return img, [targets, ]


@DATA_REGISTRY.register()
class CIFARDataModule(pl.LightningDataModule):

    @configurable
    def __init__(
        self, 
        num_classes,
        batch_size,
        eval_batch_size,
        num_workers,
        data_ratio,
        gpus,
    ):
        super().__init__()
        self.data_dir = os.path.join(hydra.utils.get_original_cwd(), "data", "CIFAR{}".format(num_classes))
        self.data_ratio = data_ratio
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.gpus = gpus
        self.augmentation = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.test_augmentation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        assert num_classes in [10, 100], "Only Support 10 and 100, where {}".format(num_classes)
        self.dataset_type = MultiTaskCIFAR10 if num_classes == 10 else MultiTaskCIFAR100
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.dataset.meta_data.classpredict.num_classes,
            "batch_size": cfg.dataset.dataloader.batch_size,
            "eval_batch_size":  cfg.dataset.eval.batch_size,
            "num_workers": cfg.dataset.dataloader.num_workers,
            "data_ratio": cfg.dataset.dataloader.data_ratio,
            "gpus": cfg.dataset.gpus
        }

    def setup(self, stage: Optional[str] = None):
        # Assign train/val dataset for use in dataloaders
        self.train_dataset = self.dataset_type(root=self.data_dir, train=True, transform=self.augmentation, download=True,
                                               ratio=self.data_ratio)
        self.val_dataset = self.dataset_type(root=self.data_dir, train=False, transform=self.test_augmentation, download=True,
                                             ratio=self.data_ratio)

    def train_dataloader(self):
        batch_size = self.batch_size
        batch_size = batch_size // self.gpus
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
            batch_size=self.eval_batch_size,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, num_workers=4)