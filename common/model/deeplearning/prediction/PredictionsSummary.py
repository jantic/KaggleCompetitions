from common.image.ImageInfo import ImageInfo
from common.model.deeplearning.prediction import PredictionInfo


class PredictionsSummary:
    def __init__(self, image_info: ImageInfo, predictions: []):
        self.__image_info = image_info
        predictions.sort(reverse=True)
        self.__predictions = predictions
        self.__class_id_to_confidence = dict()
        for prediction in predictions:
            self.__class_id_to_confidence[prediction.get_class_id()] = prediction.get_confidence()

    def get_all_predictions(self):
        return self.__predictions

    def get_top_prediction(self) -> PredictionInfo:
        return self.get_all_predictions()[0]

    def get_confidence_for_class_id(self, class_id):
        return self.__class_id_to_confidence.get(class_id)

    def get_test_id(self) -> int:
        return self.__image_info.get_image_number()

    def get_image_info(self) -> ImageInfo:
        return self.__image_info

    def __lt__(self, other):
        return self.get_top_prediction().get_confidence() < other.get_top_prediction().get_confidence()
