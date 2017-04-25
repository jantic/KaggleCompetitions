from abc import ABCMeta, abstractmethod
from keras.preprocessing import image
from PIL.Image import Image

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
    def predict(self, pilImages : [Image], id : int, details=False): raise NotImplementedError
    @abstractmethod
    def finetune(self, batches): raise NotImplementedError
    @abstractmethod
    def fit(self, batches, val_batches, nb_epoch=1): raise NotImplementedError