from common.image.ImageInfo import ImageInfo
from common.image.ModelImageConverter import ModelImageConverter
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest


class BatchImagePredictionRequestInfo:
    @staticmethod
    def get_instance(image_prediction_requests: [ImagePredictionRequest], target_image_width: int, target_image_height: int):
        test_id_to_ordered_image_infos = BatchImagePredictionRequestInfo.__generate_test_id_to_ordered_image_infos_mapping(image_prediction_requests)
        test_ids, image_infos = BatchImagePredictionRequestInfo.__generate_batch_data(test_id_to_ordered_image_infos)
        batch_pil_images = ModelImageConverter.get_all_pil_images(image_infos)
        image_array = ModelImageConverter.generate_image_array_for_prediction(batch_pil_images, target_image_width, target_image_height)
        return BatchImagePredictionRequestInfo(test_ids, image_infos, image_array)

    @staticmethod
    def __generate_test_id_to_ordered_image_infos_mapping(image_prediction_requests: [ImagePredictionRequest]) -> {}:
        test_id_to_ordered_image_infos = {}

        for image_prediction_request in image_prediction_requests:
            image_infos = image_prediction_request.get_image_infos()
            test_id = image_prediction_request.get_test_id()
            if not (test_id in test_id_to_ordered_image_infos):
                test_id_to_ordered_image_infos[test_id] = []

            test_id_to_ordered_image_infos[test_id].extend(image_infos)

        return test_id_to_ordered_image_infos

    @staticmethod
    def __generate_batch_data(test_id_to_ordered_image_infos: {}):
        batch_test_ids = []
        batch_image_infos = []

        for test_id in test_id_to_ordered_image_infos.keys():
            BatchImagePredictionRequestInfo.__prepare_data_for_batch(test_id_to_ordered_image_infos, test_id, batch_test_ids, batch_image_infos)

        return batch_test_ids, batch_image_infos

    @staticmethod
    def __prepare_data_for_batch(test_id_to_ordered_image_infos: {}, current_test_id: int, batch_test_ids: [int], batch_image_infos: [ImageInfo]):
        for image_info in test_id_to_ordered_image_infos[current_test_id]:
            batch_image_infos.append(image_info)
            batch_test_ids.append(current_test_id)

    def __init__(self, test_ids: [int], image_infos: [ImageInfo], image_array: [int]):
        self.__test_ids = test_ids
        self.__image_infos = image_infos
        self.__image_array = image_array

    def get_image_array(self):
        return self.__image_array

    def get_test_ids(self):
        return self.__test_ids

    def get_image_infos(self):
        return self.__image_infos
