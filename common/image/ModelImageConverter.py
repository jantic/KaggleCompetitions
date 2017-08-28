from keras.preprocessing import image as image_processing
from PIL.Image import Image
import PIL.Image
import numpy as np
from common.image.ImageInfo import ImageInfo


class ModelImageConverter:
    @staticmethod
    def get_all_pil_images(image_infos: [ImageInfo]) -> [Image]:
        pil_images = []

        for image_info in image_infos:
            pil_images.append(image_info.get_pil_image())

        return pil_images

    @staticmethod
    def generate_image_array_for_prediction(pil_images: [Image], width: int, height: int) -> [int]:
        resized_pil_images = []

        for pil_image in pil_images:
            resized_pil_image = ModelImageConverter.__generate_resized_pil_image(pil_image, width, height)
            resized_pil_images.append(resized_pil_image)

        batch_size = len(resized_pil_images)
        image_array = np.zeros((batch_size,) + (3, width, height), dtype=image_processing.K.floatx())

        for index in range(len(resized_pil_images)):
            pil_image = resized_pil_images[index]
            x = image_processing.img_to_array(pil_image)
            x = x.reshape(1, 3, width, height)
            image_array[index] = x

        return image_array

    @staticmethod
    def __generate_resized_pil_image(pil_image: Image, width: int, height: int) -> Image:
        # crop to maintain aspect ratio, then resize
        aspect_ratio = width / height
        cropped_width = min(int(aspect_ratio * pil_image.height), pil_image.width)
        cropped_height = min(int(pil_image.width / aspect_ratio), pil_image.height)
        x0 = int((pil_image.width - cropped_width) / 2)
        y0 = int((pil_image.height - cropped_height) / 2)
        x1 = pil_image.width - int((pil_image.width - cropped_width) / 2)
        y1 = pil_image.height - int((pil_image.height - cropped_height) / 2)
        cropped_image = pil_image.crop((x0, y0, x1, y1))
        resized_image = cropped_image.resize((width, height), PIL.Image.ANTIALIAS)
        return resized_image
