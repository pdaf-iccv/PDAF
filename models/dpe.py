import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDPMScheduler


class ResConv2d(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResConv2d, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Conv2d(n_feats , n_feats, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res

class Denoise2D(nn.Module):
    def __init__(self, n_feats = 64, n_conds=64, n_denoise_res = 5, timesteps=5):
        super(Denoise2D, self).__init__()
        self.max_period=timesteps*10
        resconv = [
            nn.Conv2d(n_feats + n_conds + 1, n_feats, 1, 1),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resconv.append(ResConv2d(n_feats))
        self.resconv=nn.Sequential(*resconv)

    def forward(self, x, t, c):
        b, _, h, w = x.shape
        t = torch.full((b, 1, h, w), t,  device=x.device, dtype=torch.long)
        t = t.float()
        t = t / self.max_period
        c = torch.cat([c, t, x], dim=1)
        
        fea = self.resconv(c)

        return fea

class DPE(nn.Module):
    def __init__(self, prior_cfg):
        super(DPE, self).__init__()
        self.n_feats = prior_cfg["n_feats"]
        self.denoiser = Denoise2D(
            n_feats=prior_cfg["n_feats"],
            n_conds=prior_cfg["n_conds"],
            n_denoise_res=prior_cfg["n_denoise_res"],
            timesteps=prior_cfg["timesteps"]
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=prior_cfg["timesteps"],
            beta_start=0.1,
            beta_end=0.99,
            prediction_type="sample"
        )
        if "scale" in prior_cfg:
            self.scale = prior_cfg["scale"]
        else:
            self.scale = 1

    def forward_ldm(self, latents, conds, num_inference_steps):

        self.noise_scheduler.set_timesteps(num_inference_steps, device=latents.device)
        
        for t in reversed(range(0, num_inference_steps)):
            pred_eps = self.denoiser(
                latents, t, conds
            )
            latents = self.noise_scheduler.step(pred_eps, t, latents)
            latents = latents.prev_sample

        return latents

    def forward(self, conds, targets=None):
        
        if targets != None:
            b, _, h, w = conds.size()
            targets_shapes = targets.shape
            noisy_latents = torch.randn(targets_shapes).to(conds.device)
            if self.scale != 1:
                noisy_latents = F.interpolate(noisy_latents, (int(h * self.scale), int(w * self.scale)))
                conds = F.interpolate(conds, (int(h * self.scale), int(w * self.scale)))

            preds = self.forward_ldm(
                noisy_latents, conds, self.noise_scheduler.config.num_train_timesteps
            )
            if self.scale != 1:
                preds = F.interpolate(preds, (h, w))
        else:
            b, _, h, w = conds.size()
            noisy_latents = torch.randn((b, self.n_feats, h, w)).to(conds.device)
            if self.scale != 1:
                noisy_latents = F.interpolate(noisy_latents, (int(h * self.scale), int(w * self.scale)))
                conds = F.interpolate(conds, (int(h * self.scale), int(w * self.scale)))
            preds = self.forward_ldm(
                noisy_latents, conds, self.noise_scheduler.config.num_train_timesteps
            )
            if self.scale != 1:
                preds = F.interpolate(preds, (h, w))
        return preds

def build_DPE(config):
    return DPE(config)