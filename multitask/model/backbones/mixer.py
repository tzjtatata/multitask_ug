import os
import hydra
import torch
from torch import nn
from torch.nn import functional as F

from torchlet.backbone import BACKBONE_REGISTRY


class MLPBlock(nn.Module):

    def __init__(self, in_plane, mlp_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_plane, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, in_plane)
        self.gelu = nn.GELU()

    def forward(self, x):
        y = self.fc1(x)
        y = self.gelu(y)
        return self.fc2(y)


class MixerLayer(nn.Module):

    def __init__(self, channel, sequence, channel_dim, sequence_dim):
        super(MixerLayer, self).__init__()
        self.channel = channel
        self.sequence = sequence
        self.sequence_dim = sequence_dim
        self.channel_dim = channel_dim
        self.ln = nn.LayerNorm(self.channel)
        self.mlp1 = MLPBlock(sequence, sequence_dim)
        self.mlp2 = MLPBlock(channel, channel_dim)

    def forward(self, x):
        # x: (bs, sequence, channel)
        y = self.ln(x)
        y_t = y.transpose(1, 2)
        y = self.mlp1(y_t)
        y = y.transpose(1, 2)
        y = x + y
        y = self.ln(y)
        y = x + self.mlp2(y)
        return y


@BACKBONE_REGISTRY.register()
class Mixer(nn.Module):
    """
        Adapt from github.com/google-research/vision_transformer/
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bb_cfg = self.cfg.backbone
        self.embd = nn.Conv2d(3, self.bb_cfg.channel, kernel_size=self.bb_cfg.patch_size, stride=self.bb_cfg.patch_size)
        self.ln = nn.LayerNorm(self.bb_cfg.channel)
        self.mlp_blocks = nn.ModuleList([])
        for _ in range(self.bb_cfg.num_blocks):
            self.mlp_blocks.append(
                MixerLayer(
                    self.bb_cfg.channel,
                    self.bb_cfg.sequence,
                    self.bb_cfg.channel_dim,
                    self.bb_cfg.sequence_dim
                )
            )

    def forward(self, x):
        y = self.embd(x)
        bs, c, h, w = y.shape
        y = y.view(bs, c, -1).transpose(1, 2)
        for i in range(self.bb_cfg.num_blocks):
            y = self.mlp_blocks[i](y)
        y = self.ln(y)
        y = torch.mean(y, dim=1, keepdim=False)
        return y

    def get_last_layer(self):
        return self.mlp_blocks[-1]


if __name__ == '__main__':

    mixer = Mixer()

