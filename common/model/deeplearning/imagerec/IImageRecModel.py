from abc import ABCMeta, abstractmethod
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from common.model.deeplearning.imagerec.ImagePredictionResult import ImagePredictionResult


# Interface
class IImageRecModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_image_width(self): raise NotImplementedError

    @abstractmethod
    def get_image_height(self): raise NotImplementedError

    @abstractmethod
    def predict(self, requests: [ImagePredictionRequest], batch_size: int, details=False) -> [ImagePredictionResult]: raise NotImplementedError

    @abstractmethod
    def refine_training(self, num_epochs: int): raise NotImplementedError

    @abstractmethod
    def get_classes(self)->list: raise NotImplementedError

