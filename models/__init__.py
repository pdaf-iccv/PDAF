from .evaluator import Evaluator

def build_evaluator(args, accelerator, cfg, val_loader):
    return Evaluator(args, accelerator, cfg, val_loader)