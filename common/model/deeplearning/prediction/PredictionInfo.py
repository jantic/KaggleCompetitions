class PredictionInfo:
    @staticmethod
    def generate_prediction_infos(confidences: [float], class_ids: [int], class_names: [str], min_confidence: float, max_confidence: float) -> []:
        prediction_infos = []
        for i in range(len(confidences)):
            raw_confidence = confidences[i]
            confidence = PredictionInfo.__get_clipped_confidence(raw_confidence, min_confidence, max_confidence)
            class_id = class_ids[i]
            class_name = class_names[i]
            prediction_info = PredictionInfo(confidence, class_id, class_name)
            prediction_infos.append(prediction_info)

        return prediction_infos

    def __init__(self, confidence: float, class_id: int, class_name: str):
        self.__confidence = confidence
        self.__class_id = class_id
        self.__class_name = class_name

    def get_confidence(self) -> float:
        return self.__confidence

    def get_class_id(self) -> int:
        return self.__class_id

    def get_class_name(self) -> str:
        return self.__class_name

    @staticmethod
    def __get_clipped_confidence(raw_confidence: float, min_confidence: float, max_confidence: float):
        if raw_confidence < min_confidence:
            return min_confidence

        if raw_confidence > max_confidence:
            return max_confidence

        return raw_confidence

    def __lt__(self, other):
        return self.get_confidence() < other.get_confidence()
