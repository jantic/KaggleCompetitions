from common.model.deeplearning.prediction import PredictionInfo


class PredictionsSummary:
    def __init__(self, testId : int, predictions : []):
        self.__testId = testId
        self.__predictions = predictions
        self.__classIdToConfidence = dict()
        for prediction in predictions:
            self.__classIdToConfidence[prediction.getClassId()] = prediction.getConfidence()

    def getAllPredictions(self):
        return self.__predictions

    def getTopPrediction(self) -> PredictionInfo:
        return self.getAllPredictions()[0]

    def getConfidenceForClassId(self, id):
        return self.__classIdToConfidence.get(id)

    def getTestId(self) -> int:
        return self.__testId