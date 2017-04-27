from abc import ABCMeta, abstractmethod
from keras.preprocessing import image
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from common.model.deeplearning.imagerec.ImagePredictionResult import ImagePredictionResult


#Interface
class IDeepLearningModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def getImageWidth(self): raise NotImplementedError
    @abstractmethod
    def getImageHeight(self): raise NotImplementedError
    @abstractmethod
    def getBatches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'): raise NotImplementedError
    @abstractmethod
    def predict(self, requests : [ImagePredictionRequest], batch_size : int, details=False) -> [ImagePredictionResult] : raise NotImplementedError
    @abstractmethod
    def finetune(self, batches): raise NotImplementedError
    @abstractmethod
    def fit(self, batches, val_batches, nb_epoch=1): raise NotImplementedError