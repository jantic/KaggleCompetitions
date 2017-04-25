from common.image.ImageInfo import ImageInfo
from common.image.ImageSplitter import ImageSplitter
from common.model.deeplearning.imagerec.IDeepLearningModel import IDeepLearningModel
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary
from common.model.deeplearning.prediction.PredictionInfo import PredictionInfo


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
        imageInfos = ImageInfo.loadImageInfosFromDirectory(testImagesPath)
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
        imageInfos.extend(ImageSplitter.getImageDividedIntoSquareQuadrants(sourceImageInfo))
        imageInfos.extend(ImageSplitter.getImageDividedIntoCrossQuadrants(sourceImageInfo))
        imageInfos.extend(ImageSplitter.getImageDividedIntoHorizontalHalves(sourceImageInfo))
        imageInfos.extend(ImageSplitter.getImageDividedIntoVerticalHalves(sourceImageInfo))
        imageInfos.extend(ImageSplitter.getImageHalfCenter(sourceImageInfo))
        testId = sourceImageInfo.getImageNumber()
        pilImages = self.__getAllPilImages(imageInfos)
        predictionSummaries = self.__model.predict(pilImages, testId)
        predictionSummaries.sort(reverse=True)
        return predictionSummaries[0]

    def __getAllPilImages(self, imageInfos : [ImageInfo]) -> [Image]:
        pilImages = []

        for imageInfo in imageInfos:
            pilImages.append(imageInfo.getPilImage())

        return pilImages

