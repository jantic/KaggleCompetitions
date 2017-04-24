from common.image.ImageInfo import ImageInfo
from common.image.ImageSplitter import ImageSplitter
from common.model.deeplearning.imagerec.IDeepLearningModel import IDeepLearningModel
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary
from common.model.deeplearning.prediction.PredictionInfo import PredictionInfo

from keras.preprocessing import image as image_processing
from PIL.Image import Image

class MasterImageClassifier:
    def __init__(self, model : IDeepLearningModel):
        self.__model = model

    def refineTraining(self, trainingImagesPath : str, training_batch_size : int, validationImagesPath : str, validation_batch_size : int, numEpochs : int):
        trainingBatches = self.__model.getBatches(trainingImagesPath, batch_size=training_batch_size)
        validationBatches = self.__model.getBatches(validationImagesPath, batch_size=validation_batch_size)
        self.__model.finetune(trainingBatches)
        self.__model.fit(trainingBatches, validationBatches, nb_epoch=numEpochs)

    def getAllPredictions(self, testImagesPath : str) -> [PredictionsSummary]:
        width = self.__model.getImageWidth()
        height = self.__model.getImageHeight()
        imageInfos = ImageInfo.loadImageInfosFromDirectory(testImagesPath, width, height)
        predictionSummaries = []

        for imageInfo in imageInfos:
            predictionSummary = self.getPredictionsForImage(imageInfo)
            predictionSummaries.append(predictionSummary)

        return predictionSummaries

    #Takes source image info, creates different versions of the same image,
    # and returns the prediction with the most confidence
    def getPredictionsForImage(self, sourceImageInfo : ImageInfo) -> PredictionsSummary:
        imageInfos = []
        imageInfos.append(sourceImageInfo)
        imageInfos.extend(ImageSplitter.getImageDividedIntoQuadrants(sourceImageInfo))
        imageInfos.extend(ImageSplitter.getImageDividedIntoHorizontalHalves(sourceImageInfo))
        imageInfos.extend(ImageSplitter.getImageDividedIntoVerticalHalves(sourceImageInfo))
        predictionSummaries = []

        for imageInfo in imageInfos:
            width = self.__model.getImageWidth()
            height = self.__model.getImageHeight()
            resizedImageInfo = ImageInfo.getResizedImageInfoInstance(width, height, imageInfo)
            predictionInfos = self.__getPredictionsForProperlySizedImage(resizedImageInfo)
            testId = imageInfo.getImageNumber()
            predictionSummary = PredictionsSummary(testId, predictionInfos)
            predictionSummaries.append(predictionSummary)

        predictionSummaries.sort(reverse=True)
        return predictionSummaries[0]


    def __getPredictionsForProperlySizedImage(self, imageInfo : ImageInfo) -> [PredictionInfo]:
        pilImage = imageInfo.getPilImage()
        finalImage = self.__generateImageArrayForPrediction(pilImage)
        predictionInfos = self.__model.predict(finalImage)
        return predictionInfos


    def __generateImageArrayForPrediction(self, pilImage : Image):
        imageArray= image_processing.img_to_array(pilImage)
        return self.__reshapeImageForPrediction(imageArray)

    def __reshapeImageForPrediction(self, imageArray : [int]):
        width = self.__model.getImageWidth()
        height = self.__model.getImageHeight()
        return imageArray.reshape(1, 3, width, height)