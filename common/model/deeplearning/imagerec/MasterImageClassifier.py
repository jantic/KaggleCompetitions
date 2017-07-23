from common.image.ImageInfo import ImageInfo
from common.image.ImageSplitter import ImageSplitter
from common.model.deeplearning.imagerec.IImageRecModel import IImageRecModel
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary
from common.math.MathUtils import MathUtils

class MasterImageClassifier:
    def __init__(self, model: IImageRecModel):
        self.__model = model

    # Takes source image info, creates different versions of the same image,
    # and returns the prediction with the most confidence
    # TODO:  Determine batch sizes automatically?  That would be nice!
    def getAllPredictions(self, testImagesPath: str, useImageSplitting: bool, batch_size: int) -> [PredictionsSummary]:
        sourceImageInfos = ImageInfo.loadImageInfosFromDirectory(testImagesPath)
        testImageInfos = MasterImageClassifier.__generateAllTestImages(sourceImageInfos, useImageSplitting)

        predictionSummaries = []
        imagesPerTestId = int(round(len(testImageInfos) / len(sourceImageInfos), 0))
        requestSize = MathUtils.lcm(batch_size, imagesPerTestId)

        while len(testImageInfos) > 0:
            batchTestImageInfos = []
            # To reduce memory footprint- only request a portion at a time that is lcm of the batch size and number of
            # test images per test id, to group same test id images together
            while len(testImageInfos) > 0 and len(batchTestImageInfos) < requestSize:
                batchTestImageInfos.append(testImageInfos.pop())

            predictionSummaries.extend(self.__getPredictionsForAllImages(batchTestImageInfos, batch_size))

        return predictionSummaries

    @staticmethod
    def __generateAllTestImages(fullImageInfos: [ImageInfo], useImageSplitting: bool):
        testImageInfos = []

        for fullImageInfo in fullImageInfos:
            testImageInfos.append(fullImageInfo)

            if useImageSplitting:
                testImageInfos.extend(ImageSplitter.getImageDividedIntoSquareQuadrants(fullImageInfo))
                testImageInfos.extend(ImageSplitter.getImageDividedIntoCrossQuadrants(fullImageInfo))
                testImageInfos.extend(ImageSplitter.getImageDividedIntoHorizontalHalves(fullImageInfo))
                testImageInfos.extend(ImageSplitter.getImageDividedIntoVerticalHalves(fullImageInfo))
                testImageInfos.extend(ImageSplitter.getImageHalfCenter(fullImageInfo))

        return testImageInfos

    def __getPredictionsForAllImages(self, imageInfos: [ImageInfo], batch_size: int) -> [PredictionsSummary]:
        requests = ImagePredictionRequest.generateInstances(imageInfos)
        results = self.__model.predict(requests, batch_size)
        finalPredictionSummaries = []

        for result in results:
            fullImagePredictionSummary = MasterImageClassifier.__getFullImagePredictionSummary(result.getPredictionSummaries())
            allPredictionSummaries = result.getPredictionSummaries()
            finalPredictionSummary = MasterImageClassifier.__generateFinalPredictionSummary(fullImagePredictionSummary, allPredictionSummaries)
            finalPredictionSummaries.append(finalPredictionSummary)

        return finalPredictionSummaries

    @staticmethod
    def __getFullImagePredictionSummary(predictionSummaries: [PredictionsSummary]) -> PredictionsSummary:
        summaryWithLargestImage = predictionSummaries[0]

        for predictionSummary in predictionSummaries:
            currentImageInfo = predictionSummary.getImageInfo()
            currentImageArea = currentImageInfo.getWidth() * currentImageInfo.getHeight()
            topImageInfo = summaryWithLargestImage.getImageInfo()
            maxImageArea = topImageInfo.getWidth() * topImageInfo.getHeight()
            if currentImageArea > maxImageArea:
                summaryWithLargestImage = predictionSummary

        return summaryWithLargestImage

    # Generates "tie-breaker" out of subimage predictions if there isn't sufficient confidence on the top prediction
    # for the full image.
    # TODO: How exactly should that threshold be determined...?  For now, using one that works for two classes.  Definitely revisit
    @staticmethod
    def __generateFinalPredictionSummary(fullImagePredictionSummary: PredictionsSummary, predictionSummaries: [PredictionsSummary]) -> PredictionsSummary:
        if MasterImageClassifier.__meetsMinConfidenceThreshold(fullImagePredictionSummary):
            return fullImagePredictionSummary

        predictionSummaries.sort(reverse=True)
        return predictionSummaries[0]

    @staticmethod
    def __meetsMinConfidenceThreshold(predictionSummary: PredictionsSummary):
        topPredictionConfidence = predictionSummary.getTopPrediction().getConfidence()
        predictions = predictionSummary.getAllPredictions()
        predictions.sort(reverse=True)
        nextPredictionConfidence = predictions[1].getConfidence()

        if nextPredictionConfidence == 0.0:
            return False

        confidenceThreshold = 3.0  # arbitrary, magic, I know

        try:
            return topPredictionConfidence / nextPredictionConfidence > confidenceThreshold
        # overflow warning- means the ratio is really large.
        except RuntimeWarning:
            return True
