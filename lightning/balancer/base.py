import torch
from torch import nn

from lightning.utils.configs import configurable


class BaseBalancer(nn.Module):

    @configurable
    def __init__(
        self, 
        task_names,
        num_tasks,
        device,
        scales=None
    ):
        """
        We Summary the common attribute which every balancer should own.
        @param optimizer: the optimizer which contains some parameters.
        @param num_task: the number of tasks.
        @param params: the parameters which will be used to calculate weights or be weights.
        """
        super(BaseBalancer, self).__init__()
        self.num_task = num_tasks
        self.task_name = task_names
        self.optimizer = None
        self.need_grad = False
        self.need_optimizer = False
        self.need_outer_optimize = False
        self.is_mgda = False
        self.is_pcgrad = False
        self.is_imtl = False
        self.use_last_layer = False
        if scales is not None:
            self.register_buffer('scales', torch.as_tensor(scales))
        self.device = device
    
    @classmethod
    def from_config(cls, cfg):
        scales = None
        task_names = [t.lower() for t in cfg.dataset.task.tasks] + [t.lower() for t in cfg.dataset.task.aux_tasks]
        if 'loss' in cfg.dataset and cfg.dataset.loss.use_scales and len(task_names) > 1:
            scales = [cfg.dataset.loss[t] for t in task_names]
        
        if cfg.dataset.name == 'celeba' and 'multi_label' in cfg.dataset.task and cfg.dataset.task.multi_label:
            num_tasks = 40
        else:
            num_tasks = len(task_names)

        # if "task_num" in cfg.dataset.task:
        #     assert len(task_names) == 1
        #     task_names = ["{}_{}".format(task_names[0], i) for i in range(cfg.dataset.task.task_num)]
        return {
            "task_names": task_names,
            "num_tasks": num_tasks,
            "scales": scales,
            'device': cfg.dataset.device
        }

    def before_bp(self):
        return

    def after_bp(self):
        return

    def after_optim(self):
        return

    def run(self, losses, **kwargs):
        """
        Weight loss use self.weights.
        Base Operation is preprocessing the losses into tensor.

        """
        if isinstance(losses, dict):
            # make sure that losses is ordered.
            ordered_losses = [losses["{}".format(k.lower())] for k in self.task_name]
        else:
            ordered_losses = losses
            # Specific for CelebA
            if isinstance(losses[0], dict):
                ordered_losses = []
                for i in range(self.num_task):
                    ordered_losses.append(losses[0][str(i)])
        ordered_losses = torch.stack(ordered_losses)
        if hasattr(self, 'scales'):
            ordered_losses = ordered_losses * self.scales
        return ordered_losses

    @property
    def weights(self):
        return self.params

    def get_weights(self):
        weights = [
            self.weights[i].item()
            for i in range(self.num_task)
        ]
        # Specific for CelebA.
        if len(self.task_name) != self.num_task:
            ret = {
                "{}_weights".format(i): weights[i]
                for i in range(self.num_task)
            }
        else:
            ret = {
                "{}_weights".format(k.lower()): weights[i]
                for i, k in enumerate(self.task_name)
            }

        # This Ratio is just calculate for two task scene.
        # In order to make sure SEMSEG task at the top, This code can not be general.
        if self.num_task == 2:
            ind = 0 if self.task_name[0] == 'SEMSEG' else 1
            ratio_ret = {
                "ss_div_dep": weights[ind] / (weights[1-ind] + 1e-8),
                "ss_div_all": weights[ind] / (weights[1-ind]+weights[ind] + 1e-8)
            }
            ret.update(ratio_ret)
        return ret


class BaseAuxBalancer(BaseBalancer):

    @configurable
    def __init__(
        self, 
        aux_tasks,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.aux_tasks = aux_tasks
    
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['aux_tasks'] = cfg.dataset.task.aux_tasks
        return ret