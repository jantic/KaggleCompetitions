import glob
import os

from keras.preprocessing import image as image_processing


class ImageInfo:
    @staticmethod
    def loadImageInfosFromDirectory(imagesDirectoryPath : str, target_size : []):
        fileExtension = "jpg"
        imagesLocator = os.path.join(imagesDirectoryPath, "*/", "*." + fileExtension)
        imagePaths = glob.glob(imagesLocator, recursive=True)
        imageInfos = []

        for imagePath in imagePaths:
            imageInfo = ImageInfo(imagePath, target_size)
            imageInfos.append(imageInfo)

        return imageInfos

    def __init__(self, imagePath, target_size):
        self.__target_size = target_size
        self.__imagePath = imagePath
        self.__imageNumber = self.__determineImageNumber(imagePath)
        self.__imageArray = self.__generateResizedImageArray(imagePath, target_size)

    def getImageNumber(self) -> int:
        return self.__imageNumber

    def getImageArray(self) -> []:
        return self.__imageArray

    def getTargetSize(self) -> []:
        return self.__target_size

    def getImagePath(self) -> str:
        return self.__imagePath

    def __determineImageNumber(self, imagePath) -> int:
        return os.path.split(imagePath)[-1][0:-4]

    def __generateResizedImageArray(self, imagePath, target_size) -> []:
        rawImage = image_processing.load_img(imagePath, target_size=target_size)
        rawImageArray = image_processing.img_to_array(rawImage)
        return rawImageArray.reshape(1, 3, target_size[0], target_size[1])