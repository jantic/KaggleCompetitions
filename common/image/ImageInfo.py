import glob
import os
import random

from keras.preprocessing import image as image_processing
import PIL.Image
from PIL.Image import Image

class ImageInfo:
    @staticmethod
    def loadImageInfosFromDirectory(imagesDirectoryPath : str, width : int, height : int):
        fileExtension = "jpg"
        imagesLocator = os.path.join(imagesDirectoryPath, "*/", "*." + fileExtension)
        imagePaths = glob.glob(imagesLocator, recursive=True)
        imageInfos = []

        for imagePath in imagePaths:
            imageInfo = ImageInfo.getInstanceForImagePath(width, height, imagePath)
            imageInfos.append(imageInfo)

        return imageInfos

    @staticmethod
    def getInstanceForImagePath(width : int, height : int, imagePath : str):
        pilImage = ImageInfo.__loadPILImageFromPath(imagePath)
        imageNumber = ImageInfo.__determineImageNumber(imagePath)
        return ImageInfo.getInstance(width, height, imageNumber, pilImage)

    @staticmethod
    def getResizedImageInfoInstance(width : int, height : int, imageInfo):
        pilImage = imageInfo.getPilImage()
        imageNumber = imageInfo.getImageNumber()
        return ImageInfo.getInstance(width, height, imageNumber, pilImage)

    @staticmethod
    def getInstance(width : int, height : int, imageNumber : int, pilImage : Image):
        resizedPilImage = ImageInfo.__generateResizedPilImage(pilImage, width, height)
        return ImageInfo(width, height, imageNumber, resizedPilImage)

    def __init__(self, width : int, height : int, imageNumber : int, pilImage : Image):
        self.__width = width
        self.__height = height
        self.__imageNumber = imageNumber
        self.__pilImage =  pilImage

    def getImageNumber(self) -> int:
        return self.__imageNumber

    def getPilImage(self) -> []:
        return self.__pilImage

    def getTargetSize(self) -> []:
        return self.__target_size

    def getImagePath(self) -> str:
        return self.__imagePath

    def getWidth(self)-> int:
        return self.__width

    def getHeight(self)-> int:
        return self.__height

    @staticmethod
    def __determineImageNumber(imagePath) -> int:
        return os.path.split(imagePath)[-1][0:-4]

    @staticmethod
    def __loadPILImageFromPath(imagePath : str) -> Image:
        return image_processing.load_img(imagePath)

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