from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
import transformers


class SFTLayer2D(nn.Module):
    def __init__(self, n_feats, out_channels):
        super(SFTLayer2D, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(n_feats, n_feats, 1)
        self.SFT_scale_conv1 = nn.Conv2d(n_feats, out_channels, 1)
        self.SFT_shift_conv0 = nn.Conv2d(n_feats, n_feats, 1)
        self.SFT_shift_conv1 = nn.Conv2d(n_feats, out_channels, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        input_shape= x[0].shape[-2:]
        cond = F.interpolate(x[1], size=input_shape, mode='bilinear', align_corners=False)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        return x[0] * scale + shift

class SFTLayer1D(nn.Module):
    def __init__(self, n_feats, out_channels):
        super(SFTLayer1D, self).__init__()
        self.SFT_scale_linear0 = nn.Linear(n_feats, n_feats)
        self.SFT_scale_linear1 = nn.Linear(n_feats, out_channels)
        self.SFT_shift_linear0 = nn.Linear(n_feats, n_feats)
        self.SFT_shift_linear1 = nn.Linear(n_feats, out_channels)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        b, c, h, w = x[0].shape
        scale = self.SFT_scale_linear1(F.leaky_relu(self.SFT_scale_linear0(x[1]), 0.1, inplace=True)).view(-1, c, 1, 1)
        shift = self.SFT_shift_linear1(F.leaky_relu(self.SFT_shift_linear0(x[1]), 0.1, inplace=True)).view(-1, c, 1, 1)
        return x[0] * scale + shift

class ResBlock_SFT(nn.Module):
    def __init__(self, n_feats, out_channels, prior_dim):
        super(ResBlock_SFT, self).__init__()
        if prior_dim == 1:
            self.sft0 = SFTLayer1D(n_feats, out_channels)
        elif prior_dim == 2:
            self.sft0 = SFTLayer2D(n_feats, out_channels)
        else:
            raise ValueError('Unexpected dimensions.')
        self.conv0 = nn.Sequential(
            nn.Conv2d(out_channels, n_feats, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feats, out_channels, 3, 1, 1),
        )

        if prior_dim == 1:
            self.sft1 = SFTLayer1D(n_feats, out_channels)
        elif prior_dim == 2:
            self.sft1 = SFTLayer2D(n_feats, out_channels)
        else:
            raise

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, n_feats, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feats, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = F.relu(self.sft0(x), inplace=True)
        fea = self.conv0(fea)
        fea = F.relu(self.sft1((fea, x[1])), inplace=True)
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions

class DCM_Model(nn.ModuleDict):

    def __init__(self, cfg, model):

        super(DCM_Model, self).__init__()
        self.model = model
        in_channels = sum([v['n_feats'] for _, v in cfg['Prior Layers'].items()])
        self.prior_insert_layers = cfg['Prior Insert Layers']
        self.prior_blocks = nn.ModuleDict(
            {
                name : ResBlock_SFT(in_channels, param["out_channels"], prior_dim=2) for name, param in self.prior_insert_layers.items()
            }
        )

    def encode(self, x, prior=None):
        features = self.model.encode(x)
        return self.fuse(features, prior)

    def fuse(self, features, prior):
        output = OrderedDict()
        if isinstance(features, transformers.modeling_outputs.BackboneOutput):
            features = {f'layer{i+1}': tensor for i, tensor in enumerate(features.feature_maps)}
        for name, feature in features.items():
            if prior != None and name in self.prior_insert_layers:
                output[name] = self.prior_blocks[name]((feature, prior))[0]
            else:
                output[name] = feature
        return output

    def decode(self, features):
        return self.model.decode(features)

    def forward(self, images):
        return self.model(images)
