import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
from torchlet.backbone import BACKBONE_REGISTRY

"""
    This File adapts from github.com/loherent/mtan
"""

# parser = argparse.ArgumentParser(description='Multi-task: Split')
# parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
# parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
# parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
# parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
# parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
# opt = parser.parse_args()


@BACKBONE_REGISTRY.register()
class SegNet(nn.Module):
    def __init__(self, cfg):
        super(SegNet, self).__init__()
        self.cfg = cfg
        # initialise network parameters
        # if opt.type == 'wide':
        #     filter = [64, 128, 256, 512, 1024]
        # else:
        #     filter = [64, 128, 256, 512, 512]
        # Without wide, we temporally use this setting.
        filter = [64, 128, 256, 512, 512]

        """
        Define encoder decoder layers.
        encoder_block = [
            conv_layr(3 -> 64),
            conv_layr(64 -> 128),
            conv_layr(128 -> 256),
            conv_layr(256 -> 512),
            conv_layr(512 -> 512),
        ]
        decoder_block = [
            conv_layr(64 -> 64),
            conv_layr(128 -> 64),
            conv_layr(256 -> 128),
            conv_layr(512 -> 256),
            conv_layr(512 -> 512),
        ]
        
        """
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        """
        Define Convolution layer.
        
        """
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task specific layers
        # we don't need head in this module.
        # self.pred_task1 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
        #                                 nn.Conv2d(in_channels=filter[0], out_channels=self.class_nb, kernel_size=1, padding=0))
        # self.pred_task2 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
        #                                 nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0))
        # self.pred_task3 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
        #                                 nn.Conv2d(in_channels=filter[0], out_channels=3, kernel_size=1, padding=0))

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # define convolutional block
    def conv_layer(self, channel):
        # if opt.type == 'deep':
        #     conv_block = nn.Sequential(
        #         nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
        #         nn.BatchNorm2d(num_features=channel[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1),
        #         nn.BatchNorm2d(num_features=channel[1]),
        #         nn.ReLU(inplace=True),
        #     )
        # else:
        #     conv_block = nn.Sequential(
        #         nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
        #         nn.BatchNorm2d(num_features=channel[1]),
        #         nn.ReLU(inplace=True)
        #     )
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
        )
        return conv_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # global shared encoder-decoder network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        return g_decoder[4][1]

