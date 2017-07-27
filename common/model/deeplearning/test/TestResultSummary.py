from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary


class TestResultSummary:
    def __init__(self, predictionSummary: PredictionsSummary, actualClassName: str):
        self.__predictionSummary = predictionSummary
        self.__actualClassName = actualClassName

    def getPredictionSummary(self) -> PredictionsSummary:
        return self.__predictionSummary

    def getActualClassName(self) -> str:
        return self.__actualClassName

    def isCorrect(self) -> bool:
        return self.__actualClassName == self.__predictionSummary.getTopPrediction().getClassName()
