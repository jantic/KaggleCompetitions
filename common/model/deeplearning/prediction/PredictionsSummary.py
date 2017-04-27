from common.image.ImageInfo import ImageInfo
from common.model.deeplearning.prediction import PredictionInfo


class PredictionsSummary:
    def __init__(self, imageInfo : ImageInfo, predictions : []):
        self.__imageInfo = imageInfo
        predictions.sort(reverse=True)
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
        return self.__imageInfo.getImageNumber()

    def getImageInfo(self) -> ImageInfo:
        return self.__imageInfo

    def __lt__(self, other):
        return self.getTopPrediction().getConfidence() < other.getTopPrediction().getConfidence()