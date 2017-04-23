from common.image.ImageInfo import ImageInfo
from common.model.deeplearning.imagerec.IDeepLearningModel import IDeepLearningModel
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary


class MasterImageClassifier:
    def __init__(self, model : IDeepLearningModel):
        self.__model = model

    def refineTraining(self, trainingImagesPath : str, training_batch_size : int, validationImagesPath : str, validation_batch_size : int, numEpochs : int):
        trainingBatches = self.__model.getBatches(trainingImagesPath, batch_size=training_batch_size)
        validationBatches = self.__model.getBatches(validationImagesPath, batch_size=validation_batch_size)
        self.__model.finetune(trainingBatches)
        self.__model.fit(trainingBatches, validationBatches, nb_epoch=numEpochs)

    def getAllPredictions(self, testImagesPath):
        width = self.__model.getImageWidth()
        height = self.__model.getImageHeight()
        imageInfos = ImageInfo.loadImageInfosFromDirectory(testImagesPath, width, height)
        predictionSummaries = []

        for imageInfo in imageInfos:
            predictionInfos = self.__getPredictionsForProperlySizedImage(imageInfo)
            testId = imageInfo.getImageNumber()
            predictionSummary = PredictionsSummary(testId, predictionInfos)
            predictionSummaries.append(predictionSummary)

        return predictionSummaries

    def getPredictionsForImage(self, imageInfo : ImageInfo):
        width = self.__model.getImageWidth()
        height = self.__model.getImageHeight()
        resizedImageInfo = ImageInfo.getResizedImageInfoInstance(width, height, imageInfo)
        return self.__getPredictionsForProperlySizedImage(resizedImageInfo)

    def __getPredictionsForProperlySizedImage(self, imageInfo : ImageInfo):
        image = imageInfo.getImageArray()
        predictionInfos = self.__model.predict(image)
        return predictionInfos
