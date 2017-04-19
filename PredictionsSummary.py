import ImageInfo
import PredictionInfo


class PredictionsSummary:
    def __init__(self, testId : int, predictions):
        self.__testId = testId
        self.__predictions = predictions

    def getAllPredictions(self):
        return self.__predictions

    def getTopPrediction(self) -> PredictionInfo:
        return self.getAllPredictions()[0]

    def getTestId(self) -> int:
        return self.__testId