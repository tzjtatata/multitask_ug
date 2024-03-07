import os
import hydra
from torch import nn
from torchvision.models.vgg import cfgs as model_archs, model_urls
from lightning.utils.configs import configurable
from torch.hub import load_state_dict_from_url

from torchlet.backbone import BACKBONE_REGISTRY


my_model_archs = {
    'RD': ['RM', 512, 512, 512, 'RM', 512, 512, 256, 'RM', 256, 256, 128, 'RM', 128, 64, 'RM', 64]
}
model_archs.update(my_model_archs)


def make_layers(model_arch, use_bn=True, in_channels=3):
    layers = []
    current_channels = in_channels
    current_stage = []
    for l in model_arch:
        if l == 'M':
            current_stage.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            layers.append(nn.Sequential(*current_stage))
            current_stage = []
        elif l == 'RM':
            if len(current_stage):
                layers.append(nn.Sequential(*current_stage))
            current_stage = []
            layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(current_channels, l, kernel_size=3, padding=1)
            if use_bn:
                # In VGG style, BN use before ReLU.
                current_stage += [conv2d, nn.BatchNorm2d(l), nn.ReLU(inplace=True)]
            else:
                current_stage += [conv2d, nn.ReLU(inplace=True)]
            current_channels = l
    if len(current_stage):
        layers.append(nn.Sequential(*current_stage))

    return layers


def generate_state_dict(model, pretrained, is_reverse=False):
    loaded = {
        k: v
        for k, v in pretrained.items()
        if k.startswith('features')
    }
    if is_reverse:
        loaded.pop('features.0.weight')
        loaded.pop('features.0.bias')
    loaded_keys = list(loaded.keys())
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if is_reverse:
        keys = keys[::-1]
    i = 0
    new_state_dict = {}
    for k in keys:
        v = state_dict[k]
        # print(i, k, v.shape, loaded[loaded_keys[i]].shape)
        if k.endswith('weight') and len(v.shape) >= 4:
            bias_k = k[:-6]+'bias'
            w, b = loaded[loaded_keys[i]], loaded[loaded_keys[i+1]]
            if is_reverse:
                w = w.transpose(0, 1)
            new_state_dict[k] = w
            # Decoder not inherient bias.
            if not is_reverse:
                assert bias_k in state_dict and state_dict[bias_k].shape == b.shape, bias_k
                new_state_dict[bias_k] = b
            i += 2
        if i >= len(loaded_keys):
            break
    # print(i)
    state_dict.update(new_state_dict)
    return state_dict


@BACKBONE_REGISTRY.register()
class AutoEncoderVGG16(nn.Module):
    """
        One of the backbone, Construct a VGG16 and a Reverse VGG16,
        Connect them, and use encoder indices to support decoder before each upsampling(unpool)
    """
    @configurable
    def __init__(
        self,
        pretrained
    ):
        super().__init__()
        self.pretrained = pretrained
        self.encoder = nn.ModuleList(make_layers(model_archs['D']))  # D type is for VGG16
        self.decoder = nn.ModuleList(make_layers(model_archs['RD'], in_channels=512))
        if self.pretrained:
            try:
                root = hydra.utils.get_original_cwd()
            except:
                root = '/data1/PycharmProjects/multitask'
            model_dir = os.path.join(root, 'pretrain_model')
            pretrained_state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir=model_dir)
            encoder_state_dict = generate_state_dict(self.encoder, pretrained_state_dict)
            decoder_state_dict = generate_state_dict(self.decoder, pretrained_state_dict, is_reverse=True)
            self.encoder.load_state_dict(encoder_state_dict)
            self.decoder.load_state_dict(decoder_state_dict)
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "pretrained": cfg.dataset.model.backbone.pretrained
        }

    def forward(self, data):
        indices = []
        x = data
        for layer in self.encoder:
            x, indice = layer(x)
            # print("Indices: {}, x: {}".format(indice.shape, x.shape))
            indices.append(indice)
        i = 1
        for layer in self.decoder:
            if isinstance(layer, nn.MaxUnpool2d):
                # print("Indices: {}, x: {}".format(indices[-i].shape, x.shape))
                x = layer(x, indices[-i])
                i += 1
            else:
                # print(layer)
                x = layer(x)
        return x

    def get_last_layer(self):
        return self.decoder[-1][0]


if __name__ == '__main__':
    # model = AutoEncoderVGG16("")
    # a = torch.randn(3, 3, 640, 480)
    # b = model(a)
    # print(b.shape)

    # Use for generate pretrained AEVGG16
    model = AutoEncoderVGG16("")
    state_dict = load_state_dict_from_url(model_urls['vgg16'])
    from multitask.model.utils import show_models
    encoder_st = generate_state_dict(model.encoder, state_dict)
    decoder_st = generate_state_dict(model.decoder, state_dict, is_reverse=True)
    model.encoder.load_state_dict(encoder_st)
    model.decoder.load_state_dict(decoder_st)
