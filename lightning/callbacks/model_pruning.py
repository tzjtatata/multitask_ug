from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.callbacks.pruning import _LayerRef 
from copy import deepcopy
from pytorch_lightning import LightningModule
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from torch import nn

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_debug

_PARAM_TUPLE = Tuple[nn.Module, str]
_PARAM_LIST = Sequence[_PARAM_TUPLE]
_MODULE_CONTAINERS = (LightningModule, nn.Sequential, nn.ModuleList, nn.ModuleDict)


class myModelPruning(ModelPruning):

    def __init__(self, pruning_fn: Union[Callable, str], parameters_to_prune: _PARAM_LIST = (), parameter_names: Optional[List[str]] = None, use_global_unstructured: bool = True, amount: Union[int, float, Callable[[int], Union[int, float]]] = 0.5, apply_pruning: Union[bool, Callable[[int], bool]] = True, make_pruning_permanent: bool = True, use_lottery_ticket_hypothesis: Union[bool, Callable[[int], bool]] = True, resample_parameters: bool = False, pruning_dim: Optional[int] = None, pruning_norm: Optional[int] = None, verbose: int = 0, prune_on_train_epoch_end: bool = True, prune_head: bool = True, prune_encoder: bool = True) -> None:
        super().__init__(pruning_fn, parameters_to_prune=parameters_to_prune, parameter_names=parameter_names, use_global_unstructured=use_global_unstructured, amount=amount, apply_pruning=apply_pruning, make_pruning_permanent=make_pruning_permanent, use_lottery_ticket_hypothesis=use_lottery_ticket_hypothesis, resample_parameters=resample_parameters, pruning_dim=pruning_dim, pruning_norm=pruning_norm, verbose=verbose, prune_on_train_epoch_end=prune_on_train_epoch_end)
        self.prune_head = prune_head
        self.prune_encoder = prune_encoder
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: LightningModule) -> None:  # type: ignore
        if self._prune_on_train_epoch_end:
            rank_zero_debug("`ModelPruning.on_train_epoch_end`. Applying pruning")
            self.save_checkpoint(trainer)
            self._run_pruning(pl_module.current_epoch)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking and not self._prune_on_train_epoch_end:
            rank_zero_debug("`ModelPruning.on_validation_epoch_end`. Applying pruning")
            self.save_checkpoint(trainer)
            self._run_pruning(pl_module.current_epoch)

    
    def save_checkpoint(self, trainer: "pl.Trainer"):
        import os
        mc = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                mc = cb
                break
        assert mc is not None, "There is No ModelCheckpoint Callbacks."
        trainer.save_checkpoint(os.path.join(mc.dirpath, f"ckpt_before_pruning{mc.FILE_EXTENSION}"))
    
    @staticmethod
    def _copy_param(new: nn.Module, old: nn.Module, name: str, postfix=None) -> None:
        if postfix is not None:
            dst = getattr(new, name+postfix)
        else:
            dst = getattr(new, name)
        src = getattr(old, name)
        if dst is None or src is None or not isinstance(dst, torch.Tensor) or not isinstance(src, torch.Tensor):
            return
        dst.data = src.data.to(dst.device)
    
    def apply_lottery_ticket_hypothesis(self) -> None:
        r"""
        Lottery ticket hypothesis algorithm (see page 2 of the paper):

            1. Randomly initialize a neural network :math:`f(x; \theta_0)` (where :math:`\theta_0 \sim \mathcal{D}_\theta`).
            2. Train the network for :math:`j` iterations, arriving at parameters :math:`\theta_j`.
            3. Prune :math:`p\%` of the parameters in :math:`\theta_j`, creating a mask :math:`m`.
            4. Reset the remaining parameters to their values in :math:`\theta_0`, creating the winning ticket :math:`f(x; m \odot \theta_0)`.

        This function implements the step 4.

        The ``resample_parameters`` argument can be used to reset the parameters with a new :math:`\theta_z \sim \mathcal{D}_\theta`
        """  # noqa: E501
        print("Using Lottery Ticket")
        assert self._original_layers is not None
        for d in self._original_layers.values():
            copy = d["data"]
            names = d["names"]
            if self._resample_parameters and hasattr(copy, "reset_parameters") and callable(copy.reset_parameters):
                copy = deepcopy(copy)  # keep the original parameters
                copy.reset_parameters()
            for i, name in names:
                new, new_name = self._parameters_to_prune[i]
                self._copy_param(new, copy, name)
                if not self._make_pruning_permanent:
                    # 我们也拷贝一下到_orig里
                    self._copy_param(new, copy, name, postfix='_orig')
    
    def on_before_accelerator_backend_setup(self, trainer: "pl.Trainer", pl_module: LightningModule) -> None:
        if self._original_layers is not None:
            return
        
        prune_keys = []
        if not self.prune_head: prune_keys.append("head")
        if not self.prune_encoder: prune_keys.append("backbone")
        if len(prune_keys) > 0:
            self._parameters_to_prune = []
            parameters = self._parameter_names or ModelPruning.PARAMETER_NAMES
            for name, m in pl_module.named_modules():
                if isinstance(m, _MODULE_CONTAINERS):
                   continue 
                for pk in prune_keys:
                    if not name.startswith(pk):
                        for p_name in parameters:
                            if getattr(m, p_name, None) is not None:
                                self._parameters_to_prune.append((m, p_name))
            print("Not Prune All parameters, only prune {}.".format(len(self._parameters_to_prune)))

        parameters_to_prune = self.sanitize_parameters_to_prune(
            pl_module, self._parameters_to_prune, parameter_names=self._parameter_names
        )

        self._parameters_to_prune = self.filter_parameters_to_prune(parameters_to_prune)

        if self._use_lottery_ticket_hypothesis:
            # group modules by id. Each entry has a copy of the initial data
            # and a list of the associated parameter names to prune
            self._original_layers = {}
            for i, (module, name) in enumerate(self._parameters_to_prune):
                id_ = id(module)
                self._original_layers.setdefault(id_, _LayerRef(data=deepcopy(module.to('cpu')), names=[]))
                self._original_layers[id_]["names"].append((i, name))