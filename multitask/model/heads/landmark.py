import torch
from torch import nn
from torch.nn import functional as F
from .base import BasicHead


class LandmarkHead(BasicHead):

    def forward(self, data):
        return self.map_func(data)

    def build_head(self):
        in_plane = self.cfg.backbone.in_planes[-1]
        return nn.Linear(in_plane, 10)  # Default 5 pts landmark prediction

    def losses(self, pred, gt):
        """
        params:
            pred: input of current head, out from self.forward()
            gt: ground truth of current task.
            reduction: the reduction mode of return loss.
        """
        batch_size = pred.shape[0]
        output = pred.reshape(batch_size, -1)
        gt = gt.reshape(batch_size, -1)
        # Why I use reduction='sum'?
        # Follow Retina Face Setting: Use Smooth L1 Loss as Landmark Regression Loss
        return F.smooth_l1_loss(output, gt)

