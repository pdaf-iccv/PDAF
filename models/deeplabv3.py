from .base import Base
from .dcm import DCM_Model
from .DeepLabV3Plus_Pytorch import _get_model


class DeepLabV3plusBase(Base):
    def get_model(self):
            
        model_name = self.cfg["Task Model"]
        num_classes = 19 # 19 for city
        output_stride = 16

        model = _get_model(model_name, num_classes, output_stride)
        model = DCM_Model(self.cfg, model)
        
        return model
    
class DeepLabV3plus(DeepLabV3plusBase):
    def __init__(self, args, cfg):
        
        self.cfg = cfg
        self.args = args
        super().__init__()

    def encode(self, x, prior=None):
        features = self.model.encode(x, prior=prior)
        return features

    def fuse(self, feature, prior):
        features = self.model.fuse(feature, prior)
        return features

    def decode(self, feature):
        outputs = self.model.decode(feature)
        return outputs

    def teacher_encode(self, x):
        feature = self.teacher.encode(x)
        return feature

    def teacher_decode(self, feature):
        return self.teacher.decode(feature)

class DeepLabV3plusLight(DeepLabV3plusBase):
    def __init__(self, args, cfg):
        
        self.cfg = cfg
        self.args = args
        super().__init__()

    def encode(self, x, prior=None):
        feature = self.teacher.encode(x)
        return feature

    def decode(self, feature):
        return self.teacher.decode(feature)

    def teacher_encode(self, x):
        feature = self.teacher.encode(x)
        return feature

    def teacher_decode(self, feature):
        return self.teacher.decode(feature)

    def fuse(self, feature, prior):
        return self.teacher.fuse(feature, prior)