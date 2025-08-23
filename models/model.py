from .deeplabv3 import DeepLabV3plus

def build_DCM_model(args, cfg):
    if "deeplabv3plus" in cfg["Task Model"]:
        deeplab = DeepLabV3plus(args, cfg)
        return deeplab
    else:
        raise ValueError('Not expected model.')

def get_metrics():
    from .DeepLabV3Plus_Pytorch import _get_metrics
    return _get_metrics()