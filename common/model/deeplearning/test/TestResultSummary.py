from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary


class TestResultSummary:
    def __init__(self, prediction_summary: PredictionsSummary, actual_class_name: str):
        self.__prediction_summary = prediction_summary
        self.__actual_class_name = actual_class_name

    def get_prediction_summary(self) -> PredictionsSummary:
        return self.__prediction_summary

    def get_actual_class_name(self) -> str:
        return self.__actual_class_name

    def is_correct(self) -> bool:
        return self.__actual_class_name == self.__prediction_summary.get_top_prediction().get_class_name()
