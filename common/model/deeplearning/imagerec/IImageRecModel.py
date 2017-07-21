from abc import ABCMeta, abstractmethod
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from common.model.deeplearning.imagerec.ImagePredictionResult import ImagePredictionResult


# Interface
class IImageRecModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def getImageWidth(self): raise NotImplementedError

    @abstractmethod
    def getImageHeight(self): raise NotImplementedError

    @abstractmethod
    def predict(self, requests: [ImagePredictionRequest], batch_size: int, details=False) -> [ImagePredictionResult]: raise NotImplementedError

    @abstractmethod
    def refineTraining(self, trainingImagesPath: str, training_batch_size: int, validationImagesPath: str, validation_batch_size: int, numEpochs: int): raise NotImplementedError
