from common.model.deeplearning.IDeepLearningModel import IDeepLearningModel
from common.image.ImageInfo import ImageInfo
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary

class ImageClassifier:
    def __init__(self, model : IDeepLearningModel, image_size : [int]):
        self.__model = model
        self.__image_size = image_size

    def refineTraining(self, trainingImagesPath : str, training_batch_size : int, validationImagesPath : str, validation_batch_size : int, numEpochs : int):
        trainingBatches = self.__model.get_batches(trainingImagesPath, batch_size=training_batch_size)
        validationBatches = self.__model.get_batches(validationImagesPath, batch_size=validation_batch_size)
        self.__model.finetune(trainingBatches)
        self.__model.fit(trainingBatches, validationBatches, nb_epoch=numEpochs)

    def getAllPredictions(self, testImagesPath):
        imageInfos = ImageInfo.loadImageInfosFromDirectory(testImagesPath, self.__image_size)
        predictionSummaries = []

        for imageInfo in imageInfos:
            predictionInfos = self.getPredictionsForImage(imageInfo)
            testId = imageInfo.getImageNumber()
            predictionSummary = PredictionsSummary(testId, predictionInfos)
            predictionSummaries.append(predictionSummary)

        return predictionSummaries

    def getPredictionsForImage(self, imageInfo : ImageInfo):
        image = imageInfo.getImageArray()
        predictionInfos = self.__model.predict(image)
        return predictionInfos
