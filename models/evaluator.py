import logging

import torch
from tqdm import tqdm

from .model import get_metrics
from .pdaf import build_PDAF

class Evaluator():
    def __init__(self, args, accelerator, cfg, val_loader):
        self.args = args
        self.cfg = cfg
        self.accelerator = accelerator
        self.model = build_PDAF(args, cfg)
        self.model, self.val_loader = self.accelerator.prepare(
            self.model, val_loader
        )

    def run(self):
        self.model.eval()
        metrics = get_metrics()
        metrics.reset()   
        for batch in tqdm(self.val_loader):

            with torch.no_grad(), torch.cuda.amp.autocast():
                images, labels, _, _ = batch
                preds = self.model.tta_forward(images, scales=self.args.scales, hflip=self.args.vhflip)
                targets = labels.cpu().numpy()
                metrics.update(targets, preds)
                
        score = metrics.get_results()
        logging.info(metrics.to_str(score))
