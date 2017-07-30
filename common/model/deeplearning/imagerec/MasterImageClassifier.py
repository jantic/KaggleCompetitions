from common.image.ImageInfo import ImageInfo
from common.image.ImageSplitter import ImageSplitter
from common.model.deeplearning.imagerec.IImageRecModel import IImageRecModel
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary
from common.math.MathUtils import MathUtils
from common.model.deeplearning.test.TestResultSummary import TestResultSummary


class MasterImageClassifier:
    def __init__(self, model: IImageRecModel):
        self.__model = model

    # Takes source image info, creates different versions of the same image,
    # and returns the prediction with the most confidence
    # TODO:  Determine batch sizes automatically?  That would be nice!

    def getAllTestResults(self, testImagesPath: str, useImageSplitting: bool, batch_size: int, className: str) -> [TestResultSummary]:
        predictionSummaries = self.getAllPredictions(testImagesPath, useImageSplitting, batch_size)
        testResultSummaries = []

        for predictionSummary in predictionSummaries:
            testResultSummary = TestResultSummary(predictionSummary, className)
            testResultSummaries.append(testResultSummary)

        return testResultSummaries

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
                testImageInfos.extend(ImageSplitter.getImageDividedIntoSquareThreeQuartersCorners(fullImageInfo))
                testImageInfos.extend(ImageSplitter.getImageDividedIntoThreeQuartersCross(fullImageInfo))
                testImageInfos.extend(ImageSplitter.getImageHalfCenter(fullImageInfo))

        return testImageInfos

    def __getPredictionsForAllImages(self, imageInfos: [ImageInfo], batch_size: int) -> [PredictionsSummary]:
        requests = ImagePredictionRequest.generateInstances(imageInfos)
        results = self.__model.predict(requests, batch_size)
        finalPredictionSummaries = []

        for result in results:
            allPredictionSummaries = result.getPredictionSummaries()
            finalPredictionSummary = MasterImageClassifier.__generateFinalPredictionSummary(allPredictionSummaries)
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
    def __generateFinalPredictionSummary(predictionSummaries: [PredictionsSummary]) -> PredictionsSummary:
        predictionSummaries.sort(reverse=True)
        return predictionSummaries[0]


