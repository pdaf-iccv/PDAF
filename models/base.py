import abc
import torch
import torch.nn.functional as F

class Base(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.model = self.get_model()
        self.model.requires_grad_(True)
        self.teacher = self.get_model()
        self.teacher.requires_grad_(False)
       
    @abc.abstractmethod
    def get_model(self):
        return NotImplemented
    
    def forward(self, x):
        return self.model(x)

    @abc.abstractmethod
    def encode(self, x, prior=None):
        return NotImplemented

    @abc.abstractmethod
    def fuse(self, feature, prior):
        return NotImplemented

    @abc.abstractmethod
    def decode(self, feature):
        return NotImplemented
    
    @abc.abstractmethod
    def teacher_encode(self, x):
        return NotImplemented
    
    @abc.abstractmethod
    def teacher_decode(self, feature):
        return NotImplemented
