from .mtnet import MultiTaskNet
from .msnet import SAUnified
from .regist_meta_arch import META_ARCH_REGISTRY
# from lightning.balancer.pcgrad import PCGrad
from ..balancer.pcgrad import PCGrad
import torch
from torch import Tensor
from typing import Optional
from torch.optim.optimizer import Optimizer


@META_ARCH_REGISTRY.register()
class PCGradNet(MultiTaskNet):

    def training_step(self, batch, batch_idx):
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

        assert not self.balancer.is_sum_loss, "PCGrad can only use when balancer.is_sum_loss=False."
        losses = self.balancer.run(losses, **addition_info)

        self.pcgrader.zero_grad()
        self.balancer.before_bp()
        self.pcgrader.pc_backward(losses['loss'])
        loss = sum(losses['loss'])

        if self.balancer.params.requires_grad:
            loss += sum(losses['regular_term'])
            loss.backward(inputs=self.balancer.params)
        self.log("total_loss", loss.item(), sync_dist=True, prog_bar=True)
        self.balancer.after_bp()
        self.pcgrader.step()
        self.balancer.after_optim()

        for k, v in self.balancer.get_weights().items():
            if k.endswith('weights'):
                self.log(k, v, sync_dist=True, prog_bar=True)
            else:
                self.log(k, v, sync_dist=True)

        return loss

    def manual_backward(self, losses: Tensor, optimizer: Optional[Optimizer] = None, *args, **kwargs) -> None:
        pass

    def configure_optimizers(self):
        optimizer_dict = super(PCGradNet, self).configure_optimizers()
        self.pcgrader = PCGrad(optimizer_dict['optimizer'])
        return optimizer_dict
