# -*- coding: utf-8 -*-
import os
import torch
import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning.model import META_ARCH_REGISTRY
from lightning.callbacks.lr_callback import ETACallback
from lightning.callbacks.model_checkpoint import build_model_checkpoint
from lightning.data.regist_dataset import DATA_REGISTRY


TRAINER_MAP = {
    "PCGrad": {"name": "PCGradNet", "type": "MA"},
    "MGDA": {"name": "MGDANet", "type": "MA"},
    "CosReg": {"name": "CosRegNet", "type": "MA"},
    "MT": {"name": "MultiTaskNet", "type": "MA"},
    "Vanden": {"name": "VandenNet", "type": "MA"},
    "VandenLabelMap": {"name": "VandenLabelMap", "type": "MA"},
    "VandenISTA": {"name": "VandenISTA", "type": "MA"},
    "MTAN": {"name": "MTANet", "type": "MA"},
    "HRNet": {"name": "MTINetwork", "type": "MA"},
    "VandenMixed": {"name": "VandenMixed", "type": "MA"},
    "AUTOL": {"name": "AutoLambdaNet", "type": "AUX"},
    "AUGAUTOL": {"name": "AugAutoLambdaNet", "type": "AUX"},
}


@hydra.main(config_path="configs", config_name='multitask')
def main(cfg: DictConfig) -> None:
    # Suitable for hydra's multi-run feature
    # Use cfg.seed for repeating the same config settings.
    # TODO: set seed randomly.
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)

    callbacks = [
        # MultipleLRCallback(cfg.dataset.solver.scheduler.type),
        LearningRateMonitor(logging_interval='epoch'),
        ETACallback()
    ]
    callbacks += build_model_checkpoint(cfg, cfg.dataset.name, cfg.dataset.task.tasks if 'eval_tasks' not in cfg.dataset.task else cfg.dataset.task.eval_tasks)

    trainer_type = cfg.dataset.trainer
    model = META_ARCH_REGISTRY.get(TRAINER_MAP[trainer_type]["name"])(cfg)
    if TRAINER_MAP[trainer_type]["type"] == "MA":
        print("Notice: You are using MA to train Model.")
    print(os.getcwd())

    if cfg.is_resume:
        path = os.environ.get('CHECKPOINT_PATH')
        print("Resume model from {}".format(path))
        # Ref: https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html
        # resume_from_checkpoint setting will automatically restores model, epoch, step, LR schedulers, apex, etc.
        trainer = pl.Trainer(
            gpus=cfg.dataset.gpus,
            check_val_every_n_epoch=cfg.dataset.eval.period,
            max_epochs=cfg.dataset.solver.max_epoch,
            callbacks=callbacks,
            resume_from_checkpoint=path
        )
    else:
        trainer = pl.Trainer(
            gpus = cfg.dataset.gpus,
            check_val_every_n_epoch = cfg.dataset.eval.period,
            max_epochs = cfg.dataset.solver.max_epoch,
            callbacks = callbacks,
            reload_dataloaders_every_n_epochs=0,
            distributed_backend='ddp' if cfg.dataset.gpus >1 else None,
            multiple_trainloader_mode='min_size'
        )
    from detectron2.utils.events import EventStorage
    with EventStorage(0) as storage:
        if TRAINER_MAP[trainer_type]["type"] == 'AUX':
            trainer.fit(model)
        else:
            datamodule = DATA_REGISTRY.get(cfg.dataset.type)(cfg)
            trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

