import glob
import os

from keras.preprocessing import image as image_processing


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
        imageArray = ImageInfo.__generateImageArrayFromPath(imagePath, width, height)
        imageNumber = ImageInfo.__determineImageNumber(imagePath)
        return ImageInfo.getInstance(width, height, imageNumber, imageArray)

    @staticmethod
    def getResizedImageInfoInstance(width : int, height : int, imageInfo):
        imageArray = imageInfo.getImageArray()
        imageNumber = imageInfo.getImageNumber()
        return ImageInfo.getInstance(width, height, imageNumber, imageArray)

    @staticmethod
    def getInstance(width : int, height : int, imageNumber : int, imageArray : [int]):
        resizedImageArray = ImageInfo.__generateResizedImageArray(imageArray, width, height)
        return ImageInfo(width, height, imageNumber, resizedImageArray)

    def __init__(self, width : int, height : int, imageNumber : int, imageArray : [int]):
        self.__width = width
        self.__height = height
        self.__imageNumber = imageNumber
        self.__imageArray =  imageArray

    def getImageNumber(self) -> int:
        return self.__imageNumber

    def getImageArray(self) -> []:
        return self.__imageArray

    def getTargetSize(self) -> []:
        return self.__target_size

    def getImagePath(self) -> str:
        return self.__imagePath

    @staticmethod
    def __determineImageNumber(imagePath) -> int:
        return os.path.split(imagePath)[-1][0:-4]

    @staticmethod
    def __generateImageArrayFromPath(imagePath : str, width : int, height : int):
        rawImage = image_processing.load_img(imagePath, target_size=[width, height])
        return ImageInfo.__generateImageArrayFromImage(rawImage, width, height)

    @staticmethod
    def __generateImageArrayFromImage(rawImage : [int], width : int, height : int):
        imageArray= image_processing.img_to_array(rawImage)
        return ImageInfo.__generateResizedImageArray(imageArray, width, height)

    @staticmethod
    def __generateResizedImageArray(rawImageArray : [int], width : int, height : int) -> [int]:
        return rawImageArray.reshape(1, 3, width, height)