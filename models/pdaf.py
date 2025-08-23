import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .common import *
from .dpe import build_DPE
from .model import build_DCM_model
from .lpe import build_Multi_LPE

def extract_ckpt(name, ckpt):
    ret = {}
    for key, value in ckpt.items():
        if name in key:
            ret[key.replace(f'{name}.', '')] = value
    return ret


class Inference:   
    def tta_forward(self, images, scales=[1.0], hflip=False, return_prod=False):        
        input_shape = images.shape[-2:]
        final_output = None
        flips = [False]
        if hflip:
            flips += [True]
        for scale in scales:
            for flip in flips:
                    
                size = (int(input_shape[0]*scale), int(input_shape[1]*scale))
                images_tmp = F.interpolate(images, size=size, mode='bilinear', align_corners=False)
                if flip:
                    images_tmp = torchvision.transforms.functional.hflip(images_tmp)
                outputs = self.forward(images_tmp, call_by_multi_forward=True)
                if flip:
                    outputs = torchvision.transforms.functional.hflip(outputs)
                outputs = F.interpolate(outputs, size=input_shape, mode='bilinear', align_corners=False)
                if final_output == None:
                    final_output = outputs
                else:
                    final_output = final_output + outputs

        if return_prod:
            return final_output / (len(flips) * len(scales))

        final_output = final_output.detach().max(dim=1)[1].cpu().numpy()
        return final_output

def prepare_cond(x_dict, prior_layers):
    conds = []
    _, _, h, w = x_dict["layer4"].size()
    for k in prior_layers.keys():
        conds.append(F.interpolate(x_dict[k], (h, w)))
    return torch.cat(conds, dim=1)

class PDAF(nn.Module, Inference):
    def __init__(self, args, cfg):
        super(PDAF, self).__init__()
        self.args = args
        self.cfg = cfg

        self.dcm_model = build_DCM_model(args, cfg)
        self.lpe = build_Multi_LPE(args,cfg) 
        self.dpe = build_DPE(cfg["DPE"])

        if args.lpe_weight is not None:
            self.lpe.load_state_dict(torch.load(args.lpe_weight, weights_only=True)) 
        if args.dcm_weight is not None:
            self.dcm_model.load_state_dict(torch.load(args.dcm_weight, weights_only=False), strict=False)
        if args.dpe_weight is not None:
            self.dpe.load_state_dict(torch.load(args.dpe_weight, weights_only=False))

    def forward(self, aug_images: torch.Tensor, call_by_multi_forward: bool=False):

        with torch.no_grad():
            lq_features = self.dcm_model.encode(aug_images)
            
        x_dict = lq_features
        conds = prepare_cond(x_dict, self.cfg["Prior Layers"])
        pred_priors = self.dpe(conds)

        aug_features = self.dcm_model.fuse(
            lq_features, 
            prior=pred_priors
        )
        
        input_shape = aug_images.shape[-2:]
        outputs = self.dcm_model.decode(aug_features) 
        outputs = F.interpolate(outputs, size=input_shape, mode='bilinear', align_corners=False)

        if not call_by_multi_forward:
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            return preds
        else:
            return outputs

def build_PDAF(args, cfg):
    return PDAF(args, cfg)
