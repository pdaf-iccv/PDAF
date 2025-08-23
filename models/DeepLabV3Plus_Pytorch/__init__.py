from .network import modeling
from .metrics import StreamSegMetrics
from .utils import ext_transforms as et

def _get_model(model_name, num_classes, output_stride):
    model = modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)
    return model

def _get_metrics():
    num_classes = 19
    metrics = StreamSegMetrics(num_classes)
    return metrics