from common.image.ImageInfo import ImageInfo
from common.model.deeplearning.imagerec.BatchImagePredictionRequestInfo import BatchImagePredictionRequestInfo
from common.model.deeplearning.prediction.PredictionInfo import PredictionInfo
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary


class ImagePredictionResult:
    @staticmethod
    def generateImagePredictionResults(batchConfidences: [float], batchRequestInfo: BatchImagePredictionRequestInfo, classes: {}):
        testIdToPredictionSummaries = ImagePredictionResult.__generateTestIdToPredictionSummaries(batchConfidences,
                                                                                                  batchRequestInfo.getTestIds(), batchRequestInfo.getImageInfos(), classes)
        imagePredictionResults = []

        for testId in testIdToPredictionSummaries.keys():
            predictionSummaries = testIdToPredictionSummaries[testId]
            imagePredictionResult = ImagePredictionResult.getInstance(testId, predictionSummaries)
            imagePredictionResults.append(imagePredictionResult)

        return imagePredictionResults

    @staticmethod
    def getInstance(testId: int, predictionSummaries: [PredictionsSummary]):
        return ImagePredictionResult(testId, predictionSummaries)

    @staticmethod
    def __generateTestIdToPredictionSummaries(batchConfidences: [float], batchTestIds: [int], batchImageInfos: [ImageInfo], classes: {}) -> {}:
        testIdToPredictionSummaries = {}

        for index in range(len(batchConfidences)):
            testId = batchTestIds[index]
            imageInfo = batchImageInfos[index]
            confidences = batchConfidences[index]
            predictionSummary = ImagePredictionResult.__generatePredictionSummary(imageInfo, confidences, classes)

            if not (testId in testIdToPredictionSummaries):
                testIdToPredictionSummaries[testId] = []

            testIdToPredictionSummaries[testId].append(predictionSummary)

        return testIdToPredictionSummaries

    @staticmethod
    def __generatePredictionSummary(imageInfo: ImageInfo, confidences: [float], classes: {}) -> PredictionsSummary:
        classIds = range(len(confidences))
        classNames = [classes[classId] for classId in classIds]
        predictionInfos = PredictionInfo.generatePredictionInfos(
            confidences, classIds, classNames, 0.0, 1.0)
        predictionSummary = PredictionsSummary(imageInfo, predictionInfos)
        return predictionSummary

    def __init__(self, testId: int, predictionSummaries: [PredictionsSummary]):
        self.__testId = testId
        self.__predictionSummaries = predictionSummaries

    def getTestId(self):
        return self.__testId

    def getPredictionSummaries(self):
        return self.__predictionSummaries
