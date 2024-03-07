import os
from typing import List
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint


DATASET_KEYMAP ={
    'nyuv2_13': [
        {
            'task': 'depth',
            'monitor': 'Depth/rmse',
            'filename': 'Depth_Best_{epoch:02d}',
            'mode': 'min'
        },
        {
            'task': 'semseg',
            'monitor': 'SemSeg/mIoU',
            'filename': 'Semseg_Best_{epoch:02d}',
            'mode': 'max'
        }
    ],
    'cityscapes': [
        {
            'task': 'depth',
            'monitor': 'depth/abs_err',
            'filename': 'depth_Best_{epoch:02d}',
            'mode': 'min'
        },
        {
            'task': 'semseg',
            'monitor': 'semseg/mIoU',
            'filename': 'semseg_Best_{epoch:02d}',
            'mode': 'max'
        }
    ],
    'loren_cityscapes': [
        {
            'task': 'depth',
            'monitor': 'depth/abs_err',
            'filename': 'depth_Best_{epoch:02d}',
            'mode': 'min'
        },
        {
            'task': 'semseg',
            'monitor': 'semseg/mIoU',
            'filename': 'semseg_Best_{epoch:02d}',
            'mode': 'max'
        },
        {
            'task': 'part_seg',
            'monitor': 'part_seg/mIoU',
            'filename': 'part_seg_Best_{epoch:02d}',
            'mode': 'max'
        }
    ],
    'cityscapes2': [
        {
            'task': 'depth',
            'monitor': 'Depth/abs_err',
            'filename': 'Depth_Best_{epoch:02d}',
            'mode': 'min'
        },
        {
            'task': 'semseg',
            'monitor': 'SemSeg/mIoU',
            'filename': 'Semseg_Best_{epoch:02d}',
            'mode': 'max'
        },
        {
            'task': 'instance',
            'monitor': 'Instance/abs_err',
            'filename': 'Instance_Best_{epoch:02d}',
            'mode': 'min'
        },
    ],
    'cifar10': [
        {
            'task': 'classpredict',
            'monitor': 'Classify/mAcc',
            'filename': 'Classify_Best_{epoch:02d}',
            'mode': 'max'
        }
    ],
    'celeba': [
        {
            'task': 'attr',
            'monitor': 'attr/mAcc',
            'filename': 'Attribute_Best_{epoch:02d}',
            'mode': 'max'
        }
    ],
    'NYUMT': [
        {
            'task': 'depth',
            'monitor': 'depth/rmse',
            'filename': 'depth_rmse_Best_{epoch:02d}',
            'mode': 'min'
        },
        {
            'task': 'semseg',
            'monitor': 'semseg/mIoU',
            'filename': 'semseg_mIoU_Best_{epoch:02d}',
            'mode': 'max'
        }
    ],
    'NYUFull': [
        {
            'task': 'depth',
            'monitor': 'depth/rmse',
            'filename': 'depth_rmse_Best_{epoch:02d}',
            'mode': 'min'
        },
        {
            'task': 'semseg',
            'monitor': 'semseg/mIoU',
            'filename': 'semseg_mIoU_Best_{epoch:02d}',
            'mode': 'max'
        }
    ],
    'NYUD': [
        {
            'task': 'depth',
            'monitor': 'depth/rmse',
            'filename': 'depth_rmse_Best_{epoch:02d}',
            'mode': 'min'
        },
    ],
    # This is cityscapes with vanden format.
    'Cityscapes': [
        {
            'task': 'city_depth',
            'monitor': 'city_depth/rmse',
            'filename': 'city_depth_rmse_Best_{epoch:02d}',
            'mode': 'min'
        },
        {
            'task': 'city_semseg',
            'monitor': 'city_semseg/mIoU',
            'filename': 'city_semseg_mIoU_Best_{epoch:02d}',
            'mode': 'max'
        }
    ]
}
DATASET_KEYMAP['cifar100'] = DATASET_KEYMAP['cifar10']


class DefaultModelCheckpoint(ModelCheckpoint):

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super(DefaultModelCheckpoint, self).on_train_start(trainer, pl_module)
        init_filepath = os.path.join(self.dirpath, f"Theta0{self.FILE_EXTENSION}")
        print("Save initial parameters to {}".format(init_filepath))
        self._save_model(trainer, init_filepath)

        # Examine: PASS
        if self.verbose:
            from pytorch_lightning.callbacks import ModelPruning
            import torch

            mp = None
            for cb in trainer.callbacks:
                if isinstance(cb, ModelPruning):
                    mp = cb
                    break

            if mp is None:
                print("No ModelPruning in Callbacks!")
                return
            origin_layer = mp._original_layers
            current_state_dict = pl_module.state_dict()
            origin_state_dict = {}
            for k, v in pl_module.named_modules():
                id_ = id(v)
                if id_ not in origin_layer:
                    continue
                m = origin_layer[id_]['data']
                for _, name in origin_layer[id_]['names']:
                    p = getattr(m, name).cpu()
                    origin_state_dict[k+'.'+name] = p
            print("Found {} paramters from pruning records".format(len(origin_state_dict)))
            for k, v in origin_state_dict.items():
                assert k in current_state_dict, "Key {} not in model state dict.".format(k)
                diff = torch.mean(v-current_state_dict[k].cpu())
                if diff != 0.0:
                    print("Params {} are not consistent, {}".format(k, diff.item()))


def build_model_checkpoint(cfg, dataset_name, tasks) -> List[Callback]:
    callbacks = [
        DefaultModelCheckpoint() if not cfg.debug.save_all_models else DefaultModelCheckpoint(save_top_k=-1, monitor='total_loss', filename='cpt_{epoch:02d}',
                                                                                              mode='min', every_n_val_epochs=1)
    ]
    if len(tasks) > 1:
        callbacks.append(
            ModelCheckpoint(
                monitor='MTL_improve',
                filename='MTL_Best_{epoch:02d}',
                mode='max',
                every_n_val_epochs=1
            )
        )
    for d in DATASET_KEYMAP[dataset_name]:
        if d['task'] in tasks:
            ckpt_monitor = ModelCheckpoint(
                monitor=d['monitor'],
                filename=d['filename'],
                mode=d['mode'],
                every_n_val_epochs=1
            )
            callbacks.append(ckpt_monitor)
    return callbacks

