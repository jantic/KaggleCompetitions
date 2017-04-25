from keras.preprocessing import image as image_processing
from PIL.Image import Image
import PIL.Image
import numpy as np

class ModelImageConverter:
    @staticmethod
    def generateImageArrayForPrediction(pilImages : [Image], width : int, height : int) -> [int]:
        resizedPilImages = []

        for pilImage in pilImages:
            resizedPilImage= ModelImageConverter.__generateResizedPilImage(pilImage, width, height)
            resizedPilImages.append(resizedPilImage)

        batch_size = len(resizedPilImages)
        imageArray = np.zeros((batch_size,) + (3, width, height), dtype=image_processing.K.floatx())

        for index in range(len(resizedPilImages)):
            pilImage = resizedPilImages[index]
            x = image_processing.img_to_array(pilImage)
            x = x.reshape(1, 3, width, height)
            imageArray[index] = x

        return imageArray

    @staticmethod
    def __generateResizedPilImage(pilImage : Image, width : int, height : int) -> Image:
        #crop to maintain aspect ratio, then resize
        aspectRatio = width/height
        croppedWidth = min(int(aspectRatio * pilImage.height), pilImage.width)
        croppedHeight = min(int(pilImage.width/aspectRatio), pilImage.height)
        x0 = int((pilImage.width - croppedWidth)/2)
        y0 = int((pilImage.height - croppedHeight)/2)
        x1 = pilImage.width - int((pilImage.width - croppedWidth)/2)
        y1 = pilImage.height - int((pilImage.height - croppedHeight)/2)
        croppedImage = pilImage.crop((x0, y0, x1, y1))
        resizedImage = croppedImage.resize((width, height), PIL.Image.ANTIALIAS)
        return resizedImage