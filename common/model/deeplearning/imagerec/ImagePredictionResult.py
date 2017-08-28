from common.image.ImageInfo import ImageInfo
from common.model.deeplearning.imagerec.BatchImagePredictionRequestInfo import BatchImagePredictionRequestInfo
from common.model.deeplearning.prediction.PredictionInfo import PredictionInfo
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary


class ImagePredictionResult:
    @staticmethod
    def generate_image_prediction_results(batch_confidences: [float], batch_request_info: BatchImagePredictionRequestInfo, classes: {}):
        test_id_to_prediction_summaries = ImagePredictionResult.__generate_test_id_to_prediction_summaries(batch_confidences,
                                                                                                           batch_request_info.get_test_ids(), batch_request_info.get_image_infos(),
                                                                                                           classes)
        image_prediction_results = []

        for test_id in test_id_to_prediction_summaries.keys():
            prediction_summaries = test_id_to_prediction_summaries[test_id]
            image_prediction_result = ImagePredictionResult.get_instance(test_id, prediction_summaries)
            image_prediction_results.append(image_prediction_result)

        return image_prediction_results

    @staticmethod
    def get_instance(test_id: int, prediction_summaries: [PredictionsSummary]):
        return ImagePredictionResult(test_id, prediction_summaries)

    @staticmethod
    def __generate_test_id_to_prediction_summaries(batch_confidences: [float], batch_test_ids: [int], batch_image_infos: [ImageInfo], classes: {}) -> {}:
        test_id_to_prediction_summaries = {}

        for index in range(len(batch_confidences)):
            test_id = batch_test_ids[index]
            image_info = batch_image_infos[index]
            confidences = batch_confidences[index]
            prediction_summary = ImagePredictionResult.__generate_prediction_summary(image_info, confidences, classes)

            if not (test_id in test_id_to_prediction_summaries):
                test_id_to_prediction_summaries[test_id] = []

            test_id_to_prediction_summaries[test_id].append(prediction_summary)

        return test_id_to_prediction_summaries

    @staticmethod
    def __generate_prediction_summary(image_info: ImageInfo, confidences: [float], classes: {}) -> PredictionsSummary:
        class_ids = range(len(confidences))
        class_names = [classes[class_id] for class_id in class_ids]
        prediction_infos = PredictionInfo.generate_prediction_infos(
            confidences, class_ids, class_names, 0.0, 1.0)
        prediction_summary = PredictionsSummary(image_info, prediction_infos)
        return prediction_summary

    def __init__(self, test_id: int, prediction_summaries: [PredictionsSummary]):
        self.__test_id = test_id
        self.__prediction_summaries = prediction_summaries

    def get_test_id(self):
        return self.__test_id

    def get_prediction_summaries(self):
        return self.__prediction_summaries
