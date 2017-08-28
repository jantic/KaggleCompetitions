from common.image.ImageInfo import ImageInfo
from common.image.ImageSplitter import ImageSplitter
from common.model.deeplearning.imagerec.IImageRecModel import IImageRecModel
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary
from common.math.MathUtils import MathUtils
from common.model.deeplearning.test.TestResultSummary import TestResultSummary


class MasterImageClassifier:
    def __init__(self, model: IImageRecModel):
        self.__model = model

    # Takes source image info, creates different versions of the same image,
    # and returns the prediction with the most confidence
    # TODO:  Determine batch sizes automatically?  That would be nice!

    def get_all_test_results(self, test_images_path: str, use_image_splitting: bool, batch_size: int, class_name: str) -> [TestResultSummary]:
        prediction_summaries = self.get_all_predictions(test_images_path, use_image_splitting, batch_size)
        test_result_summaries = []

        for prediction_summary in prediction_summaries:
            test_result_summary = TestResultSummary(prediction_summary, class_name)
            test_result_summaries.append(test_result_summary)

        return test_result_summaries

    def get_all_predictions(self, test_images_path: str, use_image_splitting: bool, batch_size: int) -> [PredictionsSummary]:
        source_image_infos = ImageInfo.load_image_infos_from_directory(test_images_path)
        test_image_infos = MasterImageClassifier.__generate_all_test_images(source_image_infos, use_image_splitting)

        prediction_summaries = []
        images_per_test_id = int(round(len(test_image_infos) / len(source_image_infos), 0))
        request_size = MathUtils.lcm(batch_size, images_per_test_id)

        while len(test_image_infos) > 0:
            batch_test_image_infos = []
            # To reduce memory footprint- only request a portion at a time that is lcm of the batch size and number of
            # test images per test id, to group same test id images together
            while len(test_image_infos) > 0 and len(batch_test_image_infos) < request_size:
                batch_test_image_infos.append(test_image_infos.pop())

            prediction_summaries.extend(self.__get_predictions_for_all_images(batch_test_image_infos, batch_size))

        return prediction_summaries

    @staticmethod
    def __generate_all_test_images(full_image_infos: [ImageInfo], use_image_splitting: bool):
        test_image_infos = []

        for full_image_info in full_image_infos:
            test_image_infos.append(full_image_info)

            if use_image_splitting:
                test_image_infos.extend(ImageSplitter.get_image_divided_into_square_quadrants(full_image_info))
                test_image_infos.extend(ImageSplitter.get_image_divided_into_cross_quadrants(full_image_info))
                test_image_infos.extend(ImageSplitter.get_image_divided_into_horizontal_halves(full_image_info))
                test_image_infos.extend(ImageSplitter.get_image_divided_into_vertical_halves(full_image_info))
                test_image_infos.extend(ImageSplitter.get_image_divided_into_square_three_quarters_corners(full_image_info))
                test_image_infos.extend(ImageSplitter.get_image_divided_into_three_quarters_cross(full_image_info))
                test_image_infos.extend(ImageSplitter.get_image_half_center(full_image_info))

        return test_image_infos

    def __get_predictions_for_all_images(self, image_infos: [ImageInfo], batch_size: int) -> [PredictionsSummary]:
        requests = ImagePredictionRequest.generate_instances(image_infos)
        results = self.__model.predict(requests, batch_size)
        final_prediction_summaries = []

        for result in results:
            all_prediction_summaries = result.get_prediction_summaries()
            final_prediction_summary = MasterImageClassifier.__generate_final_prediction_summary(all_prediction_summaries)
            final_prediction_summaries.append(final_prediction_summary)

        return final_prediction_summaries

    @staticmethod
    def __get_full_image_prediction_summary(prediction_summaries: [PredictionsSummary]) -> PredictionsSummary:
        summary_with_largest_image = prediction_summaries[0]

        for prediction_summary in prediction_summaries:
            current_image_info = prediction_summary.get_image_info()
            current_image_area = current_image_info.get_width() * current_image_info.get_height()
            top_image_info = summary_with_largest_image.get_image_info()
            max_image_area = top_image_info.get_width() * top_image_info.get_height()
            if current_image_area > max_image_area:
                summary_with_largest_image = prediction_summary

        return summary_with_largest_image

    # Generates "tie-breaker" out of subimage predictions if there isn't sufficient confidence on the top prediction
    # for the full image.
    # TODO: How exactly should that threshold be determined...?  For now, using one that works for two classes.  Definitely revisit
    @staticmethod
    def __generate_final_prediction_summary(prediction_summaries: [PredictionsSummary]) -> PredictionsSummary:
        prediction_summaries.sort(reverse=True)
        return prediction_summaries[0]


