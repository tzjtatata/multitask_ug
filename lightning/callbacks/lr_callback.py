import logging
import time
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import MultiStepLR
from typing import Optional, Any
from pytorch_lightning.utilities.types import STEP_OUTPUT


class MultipleLRCallback(Callback):

    def __init__(self, scheduler_type):
        super(MultipleLRCallback, self).__init__()
        self.scheduler_type = scheduler_type
        self.on_step = False
        # self.on_step = True if scheduler_type in ['polynomial'] else False

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.scheduler_type == 'polynomial':
            lr_scheduler = trainer.lr_schedulers[0]['scheduler']
            max_steps = trainer.max_epochs
            print("Set Max Steps as {}".format(max_steps))
            lr_scheduler.set_max_steps(max_steps)

    def on_train_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: Optional = None
    ) -> None:
        if self.on_step:
            return
        self.do_step(trainer)

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.on_step:
            return
        self.do_step(trainer)

    def do_step(self, trainer):
        lr_schedulers = trainer.lr_schedulers
        for lr_dict in lr_schedulers:
            # scheduler is a dict.
            scheduler = lr_dict['scheduler']
            scheduler.step()


class ETACallback(Callback):

    def __init__(self):
        super(ETACallback, self).__init__()
        self.epoch_count = 0
        self.epoch_start_time = 0
        self.epoch_end_time = 0

    def reset(self):
        self.epoch_count = 0
        self.epoch_start_time = time.time()
        self.epoch_end_time = None

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.reset()

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.epoch_count += 1

    def on_train_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: Optional = None
    ) -> None:
        self.epoch_end_time = time.time()
        mean_used_time = (self.epoch_end_time - self.epoch_start_time) / self.epoch_count
        print(mean_used_time)
        max_epoch = trainer.max_epochs
        time_stamp = datetime.fromtimestamp(self.epoch_end_time + (max_epoch - trainer.current_epoch) * mean_used_time)
        print("Now is {}, Training will end in {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), time_stamp.strftime("%Y-%m-%d %H:%M:%S")))



