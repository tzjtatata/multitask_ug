# Adapted from: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/models/models.py

import torch
import torch.nn as nn


# pyramid pooling, bilinear upsample
class SegmentationDecoder(nn.Module):

    def __init__(self, num_class=21, fc_dim=2048, pool_scales=(1, 2, 3, 6), task_type='C', hidden_size=512):
        super(SegmentationDecoder, self).__init__()

        self.task_type = task_type
        self.hidden_size = hidden_size
        print("Now Use Head with {} hidden".format(hidden_size))

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, self.hidden_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_size),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*self.hidden_size, self.hidden_size,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_size, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        # print(conv_out.shape)
        conv5 = conv_out

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=True))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        # if self.task_type == 'C':
        #     x = nn.functional.log_softmax(x, dim=1)
        return x

