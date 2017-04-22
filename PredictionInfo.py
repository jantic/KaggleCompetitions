
class PredictionInfo:
    @staticmethod
    def generatePredictionInfos(confidences : [float], classIds : [int], classNames : [str]):
        predictionInfos = []
        for i in range(len(confidences)):
            confidence = confidences[i]
            classId = classIds[i]
            className = classNames[i]
            predictionInfo = PredictionInfo(confidence, classId, className)
            predictionInfos.append(predictionInfo)

        return predictionInfos

    def __init__(self, confidence: float, classId : int, className : str):
        self.__confidence = confidence
        self.__classId = classId
        self.__className = className

    def getConfidence(self) -> float:
        return self.__confidence

    def getClassId(self) -> int:
        return self.__classId

    def getClassName(self) -> str:
        return self.__className

    def __lt__(self, other):
        return self.getConfidence() < other.getConfidence()