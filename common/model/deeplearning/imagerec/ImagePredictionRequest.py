from common.image.ImageInfo import ImageInfo


class ImagePredictionRequest:
    @staticmethod
    def generate_instances(image_infos: [ImageInfo]) -> []:
        grouped_image_infos = {}

        for image_info in image_infos:
            test_id = image_info.get_image_number()
            if not (test_id in grouped_image_infos):
                grouped_image_infos[test_id] = []
            grouped_image_infos[test_id].append(image_info)

        requests = []

        for image_info_group in grouped_image_infos.values():
            request = ImagePredictionRequest(image_info_group)
            requests.append(request)

        return requests

    def __init__(self, image_infos: [ImageInfo]):
        self.__image_infos = image_infos
        self.__test_id = image_infos[0].get_image_number()

    def get_image_infos(self):
        return self.__image_infos

    def get_test_id(self):
        return self.__test_id
