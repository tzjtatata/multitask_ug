import numpy as np
from .mtnet import MultiTaskNet
from lightning.utils.configs import configurable
from .regist_meta_arch import META_ARCH_REGISTRY
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from lightning.model.lr_scheduler import PolynomialLR


class CosReg(nn.Module):

    def __init__(self, regular_weight, m) -> None:
        super().__init__()
        self.regular_weight = regular_weight
        self.param_list = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(m.weight), requires_grad=True)
                for i in range(2)
            ]
        )
        self.update_param = False

        def bp_hook(grad):
            if self.update_param:
                tmp = self.param_list[0].detach() + self.param_list[1].detach()
                # print(torch.norm(tmp.flatten(), dim=0))
                return tmp
            return grad

        m.weight.register_hook(bp_hook)
    
    def get_regular_term(self, grads):
        for i in range(len(grads)):
            self.param_list[i].data = grads[i]
        regular_term = self.regular_weight * (torch.cosine_similarity(self.param_list[0].flatten(), self.param_list[1].flatten(), dim=0) ** 2)
        self.update_param = True
        return regular_term
    
    def get_cos_metric(self):
        return torch.cosine_similarity(self.param_list[0].detach().flatten(), self.param_list[1].detach().flatten(), dim=0)
    
    def get_sum_norms(self):
        return torch.norm(self.param_list[0].detach().flatten()+ self.param_list[1].detach().flatten(), dim=0).item()

@META_ARCH_REGISTRY.register()
class CosRegNet(MultiTaskNet):

    @configurable
    def __init__(
        self, 
        regular_weight,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.regular_weight = regular_weight
        self.cosreg = CosReg(self.regular_weight, self.backbone.get_last_layer())
    
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['regular_weight'] = cfg.balancer.alpha
        return ret
    
    def get_param_groups(self):
        heads = []
        backbones = []
        cosreg = []
        rest = []
        rest_k = []
        for k, v in self.named_parameters():
            if k.startswith('head'):
                heads.append(v)
                continue
            if k.startswith('backbone'):
                backbones.append(v)
                continue
            if k.startswith('cosreg'):
                cosreg.append(v)
                continue
            rest.append(v)
            rest_k.append(k)
        if len(rest_k) > 0:
            print("Not use following parameter in Main Optimizer:", rest_k)

        return [
            {'params': backbones, 'type': 'backbone'},
            {'params': heads, 'type': 'head'},
            {'params': cosreg, 'type': 'cosreg'}
        ]
    
    def configure_optimizers(self):
        optimizer_type = self.optimizer_cfg['type']
        param_groups = self.get_param_groups()
        
        if optimizer_type == 'ADAM':
            main_opt = torch.optim.Adam(
                param_groups[:2],
                lr=self.optimizer_cfg["base_lr"],
                betas=(0.9, 0.999),
                weight_decay=self.optimizer_cfg["weight_decay"],
            )
            cosreg_opt = torch.optim.Adam(
                param_groups[2:],
                lr=self.optimizer_cfg["base_lr"],
                betas=(0.9, 0.999),
                weight_decay=self.optimizer_cfg["weight_decay"],
            )
        elif optimizer_type == 'SGD':
            main_opt = torch.optim.SGD(
                param_groups[:2],
                lr=self.optimizer_cfg["base_lr"],
                momentum=self.optimizer_cfg["momentum"],
                nesterov=self.optimizer_cfg["nesterov"],
                weight_decay=self.optimizer_cfg["weight_decay"]
            )
            cosreg_opt = torch.optim.SGD(
                param_groups[2:],
                lr=self.optimizer_cfg["base_lr"],
                momentum=self.optimizer_cfg["momentum"],
                nesterov=self.optimizer_cfg["nesterov"],
                weight_decay=self.optimizer_cfg["weight_decay"]
            )
        else:
            raise NotImplementedError(
                "Not Support Optimizer type {}".format(optimizer_type))
        scheduler_type = self.scheduler_cfg["type"]
        ret = [{'optimizer': main_opt}, {'optimizer': cosreg_opt}]
        for r in ret:
            if scheduler_type == 'MultiStep':
                milestones = self.scheduler_cfg["milestones"]
                if isinstance(milestones, int):
                    milestones = range(
                        0, self.max_epoch, milestones)
                    print("milestones is {}".format(milestones))
                lr_scheduler = MultiStepLR(
                    r["optimizer"],
                    milestones=milestones,
                    gamma=self.scheduler_cfg["gamma"]
                )
            elif scheduler_type == 'polynomial':
                lr_scheduler = PolynomialLR(
                    r["optimizer"],
                    power=self.scheduler_cfg["power"]
                )
            else:
                raise NotImplementedError(
                    "Not Support Scheduler type {}".format(scheduler_type))
            r['lr_scheduler'] = {'scheduler': lr_scheduler}
        return tuple(ret)

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        img, targets = batch

        preds, shared = self(img)
        # print(img.dtype, preds[0].dtype, preds[1].dtype) # To This step, data.dtype is float32
        losses = [
            self.heads[h].losses(preds[i], targets[i])
            for i, h in enumerate(self.task_names)
        ]
        # print(losses[0].dtype, losses[1].dtype)
        addition_info = {}

        for i, t_name in enumerate(self.task_names):
            self.log("{}_loss".format(t_name),
                     losses[i], prog_bar=True, sync_dist=True)
            if self.use_kpi:
                self.log("{}_kpi".format(t_name), addition_info['kpi'][i])

        # Backward for backbone
        cos_opt = self.optimizers()[1]
        cos_opt.zero_grad()
        grads = []
        for l in losses:
            g = torch.autograd.grad(l, self.backbone.get_last_layer().weight, retain_graph=True)[0]
            grads.append(g)
        regular_term = self.cosreg.get_regular_term(grads)
        self.log("regular_term", regular_term.item())
        regular_term.backward()
        # print("before step, last_weights:", self.backbone.get_last_layer().weight[0, 0])
        # print("before step, grad0:", self.cosreg.param_list[0][0, 0])
        # print("before step, cos:", self.cosreg.get_cos_metric())
        cos_opt.step()
        # print(self.cosreg.get_sum_norms())
        # print("after step, last_weights:", self.backbone.get_last_layer().weight[0, 0])
        # print("after step, grad0:", self.cosreg.param_list[0][0, 0])
        # print("after step, cos:", self.cosreg.get_cos_metric())

        loss = self.balancer.run(losses, **addition_info)
        self.log("total_loss", loss, sync_dist=True, prog_bar=True)
        
        opt = self.optimizers()[0]
        opt.zero_grad()
        self.balancer.before_bp()
        self.manual_backward(loss)
        self.cosreg.update_param = False
        # print("After CosReg, ", torch.norm(self.backbone.get_last_layer().weight.grad.detach().flatten(), dim=0))
        self.balancer.after_bp()
        opt.step()
        self.balancer.after_optim()

        self.log_dict(self.balancer.get_weights(), sync_dist=True)
        return loss

