import torch
import torch.nn as nn
from multipledispatch import dispatch

from .common import ResBlock, default_conv


class LPE(nn.Module):
    def __init__(self, param):
        super(LPE, self).__init__()
        in_channels = param["in_channels"]
        n_feats = param["n_feats"]
        n_encoder_res = param["n_encoder_res"]
        bn = param["bn"]
        addition = param["addition"]
        self.ps = param["ps"]

        if self.ps:

            self.pixel_unshuffle = nn.PixelUnshuffle(self.ps)
            E1 = [
                nn.Conv2d((in_channels + addition) * self.ps ** 2, n_feats, 3, 1, 1),
                nn.LeakyReLU(0.1, True)
            ]
        else:
            E1 = [
                nn.Conv2d(in_channels*2 + addition, n_feats, 3, 1, 1),
                nn.LeakyReLU(0.1, True)
            ]
        E2 =[
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E = E1 + E2
        if bn:
            E += [
                nn.BatchNorm2d(n_feats, affine=False),  # then normalize them to have mean 0 and std 1
            ]
        self.E = nn.Sequential(
            *E
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1),  
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1),
            nn.ReLU(),
        )
    def encode(self, x):
        fea1 = self.E(x)
        return self.enc_out_1(fea1), self.enc_out_2(fea1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        return eps.mul(std).add_(mu)

    @dispatch(torch.Tensor, torch.Tensor)
    def forward(self, aux, src):
        if self.ps:
            aux = self.pixel_unshuffle(aux)
            src = self.pixel_unshuffle(src)
        x = torch.cat((aux, src), dim=1)
        
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

class Multi_LPE(nn.Module):
    def __init__(self, args, cfg):
        super(Multi_LPE, self).__init__()
        self.branches = nn.ModuleDict(
            {
                name: build_LPE(
                    param
                ) for name, param in cfg["Prior Layers"].items()
            }
        )
    def forward(self, x_dict):
        output_dict = {}
        for name in self.branches.keys():
            output_dict[name] = self.branches[name](x_dict[name])
        return output_dict

def build_Multi_LPE(args, cfg):
    return Multi_LPE(
        args, cfg
    )

def build_LPE(param):
    return LPE(
        param
    )
