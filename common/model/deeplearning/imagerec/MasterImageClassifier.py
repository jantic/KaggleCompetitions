from common.image.ImageInfo import ImageInfo
from common.image.ImageSplitter import ImageSplitter
from common.model.deeplearning.imagerec.IImageRecModel import IImageRecModel
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary

from PIL.Image import Image

class MasterImageClassifier:
    def __init__(self, model : IImageRecModel):
        self.__model = model

    def refineTraining(self, trainingImagesPath : str, training_batch_size : int, validationImagesPath : str, validation_batch_size : int, numEpochs : int):
        trainingBatches = self.__model.getBatches(trainingImagesPath, batch_size=training_batch_size)
        validationBatches = self.__model.getBatches(validationImagesPath, batch_size=validation_batch_size)
        self.__model.finetune(trainingBatches)
        self.__model.fit(trainingBatches, validationBatches, nb_epoch=numEpochs)

    #Takes source image info, creates different versions of the same image,
    # and returns the prediction with the most confidence
    # TODO:  Determine batch sizes automatically?  That would be nice!
    def getAllPredictions(self, testImagesPath : str, batch_size : int) -> [PredictionsSummary]:
        sourceImageInfos = ImageInfo.loadImageInfosFromDirectory(testImagesPath)
        testImageInfos = self.__generateAllTestImages(sourceImageInfos)

        predictionSummaries = []

        #To reduce memory footprint- only request one batch at a time
        while len(testImageInfos) > 0:
            batchTestImageInfos = []
            while len(testImageInfos) > 0 and len(batchTestImageInfos) < batch_size:
                batchTestImageInfos.append(testImageInfos.pop())

            predictionSummaries.extend(self.__getPredictionsForAllImages(batchTestImageInfos, batch_size))

        return predictionSummaries

    def __generateAllTestImages(self, fullImageInfos : [ImageInfo]):
        testImageInfos = []

        for fullImageInfo in fullImageInfos:
            testImageInfos.append(fullImageInfo)
            testImageInfos.extend(ImageSplitter.getImageDividedIntoSquareQuadrants(fullImageInfo))
            testImageInfos.extend(ImageSplitter.getImageDividedIntoCrossQuadrants(fullImageInfo))
            testImageInfos.extend(ImageSplitter.getImageDividedIntoHorizontalHalves(fullImageInfo))
            testImageInfos.extend(ImageSplitter.getImageDividedIntoVerticalHalves(fullImageInfo))
            testImageInfos.extend(ImageSplitter.getImageHalfCenter(fullImageInfo))

        return testImageInfos

    def __getPredictionsForAllImages(self, imageInfos : [ImageInfo], batch_size : int) -> [PredictionsSummary]:
        requests = ImagePredictionRequest.generateInstances(imageInfos)
        results = self.__model.predict(requests, batch_size)
        finalPredictionSummaries = []

        for result in results:
            fullImagePredictionSummary = self.__getFullImagePredictionSummary(result.getPredictionSummaries())
            allPredictionSummaries = result.getPredictionSummaries()
            finalPredictionSummary = self.__generateFinalPredictionSummary(fullImagePredictionSummary, allPredictionSummaries)
            finalPredictionSummaries.append(finalPredictionSummary)

        return finalPredictionSummaries


    def __getFullImagePredictionSummary(self, predictionSummaries : [PredictionsSummary]) -> PredictionsSummary:
        summaryWithLargestImage = predictionSummaries[0]

        for predictionSummary in predictionSummaries:
            currentImageInfo = predictionSummary.getImageInfo()
            currentImageArea = currentImageInfo.getWidth() * currentImageInfo.getHeight()
            topImageInfo = summaryWithLargestImage.getImageInfo()
            maxImageArea = topImageInfo.getWidth() * topImageInfo.getHeight()
            if(currentImageArea > maxImageArea):
                summaryWithLargestImage = predictionSummary

        return summaryWithLargestImage

    def __getAllPilImages(self, imageInfos : [ImageInfo]) -> [Image]:
        pilImages = []

        for imageInfo in imageInfos:
            pilImages.append(imageInfo.getPilImage())

        return pilImages

    #Generates "tie-breaker" out of subimage predictions if there isn't sufficient confidence on the top prediction
    #for the full image.
    #TODO: How exactly should that threshold be determined...?  For now, using one that works for two classes.  Definitely revisit
    def __generateFinalPredictionSummary(self, fullImagePredictionSummary : PredictionsSummary, predictionSummaries : [PredictionsSummary]) -> PredictionsSummary:
        if(self.__meetsMinConfidenceThreshold(fullImagePredictionSummary)):
            return fullImagePredictionSummary

        predictionSummaries.sort(reverse=True)
        return predictionSummaries[0]

    def __meetsMinConfidenceThreshold(self, predictionSummary : PredictionsSummary):
        topPredictionConfidence = predictionSummary.getTopPrediction().getConfidence()
        predictions = predictionSummary.getAllPredictions()
        predictions.sort(reverse=True)
        nextPredictionConfidence = predictions[1].getConfidence()
        confidenceThreshold = 3.0 #arbitrary, magic, I know
        return topPredictionConfidence/nextPredictionConfidence > confidenceThreshold

