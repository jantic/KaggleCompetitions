from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary

class ImagePredictionResult:

    @staticmethod
    def getInstance(testId : int, predictionSummaries : [PredictionsSummary]):
        return ImagePredictionResult(testId, predictionSummaries)

    def __init__(self, testId : int, predictionSummaries : [PredictionsSummary]):
        self.__testId = testId
        self.__predictionSummaries = predictionSummaries


    def getTestId(self):
        return self.__testId

    def getPredictionSummaries(self):
        return self.__predictionSummaries

