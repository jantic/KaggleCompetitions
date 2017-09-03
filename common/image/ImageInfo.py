import glob
import os

from keras.preprocessing import image as image_processing
from PIL.Image import Image

from common.image.CropBox import CropBox


class ImageInfo:
    @staticmethod
    def load_image_infos_from_directory(images_directory_path: str):
        file_extension = "jpg"
        images_locator = os.path.join(images_directory_path+"/**", "*." + file_extension)
        image_paths = glob.glob(images_locator, recursive=True)
        image_infos = []

        for image_path in image_paths:
            image_info = ImageInfo.get_instance_for_image_path(image_path)
            image_infos.append(image_info)

        return image_infos

    @staticmethod
    def get_instance_for_image_path(image_path: str):
        image_number = ImageInfo.__determine_image_number(image_path)
        return ImageInfo.get_instance(image_number, image_path)

    @staticmethod
    def get_instance(image_number: int, image_path: str, crop_box: CropBox = None):
        return ImageInfo(image_number, image_path, crop_box)

    def __init__(self, image_number: int, image_path: str, crop_box: CropBox = None):
        self.__image_number = image_number
        self.__image_path = image_path
        self.__crop_box = crop_box
        # Just using for dimension info, then discarding to preserve memory
        pil_image = self.get_pil_image()
        self.__width = pil_image.width
        self.__height = pil_image.height

    def get_image_number(self) -> int:
        return self.__image_number

    # lazy loading, to prevent huge amounts of memory being used
    def get_pil_image(self) -> Image:
        original_pil_image = ImageInfo.__load_pil_image_from_path(self.__image_path)

        if self.__crop_box is None:
            return original_pil_image

        return ImageInfo.__get_pil_image_portion(original_pil_image, self.__crop_box)

    def get_image_path(self) -> str:
        return self.__image_path

    def get_width(self) -> int:
        return self.__width

    def get_height(self) -> int:
        return self.__height

    @staticmethod
    def __determine_image_number(image_path) -> int:
        return os.path.split(image_path)[-1][0:-4]

    @staticmethod
    def __load_pil_image_from_path(image_path: str) -> Image:
        return image_processing.load_img(image_path)

    @staticmethod
    def __get_pil_image_portion(source_pil_image: Image, crop_box: CropBox) -> Image:
        return source_pil_image.crop((crop_box.get_begin_x(), crop_box.get_begin_y(), crop_box.get_end_x(), crop_box.get_end_y()))
