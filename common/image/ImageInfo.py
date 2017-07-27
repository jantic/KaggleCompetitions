import glob
import os

from keras.preprocessing import image as image_processing
from PIL.Image import Image

from common.image.CropBox import CropBox


class ImageInfo:
    @staticmethod
    def loadImageInfosFromDirectory(imagesDirectoryPath: str):
        fileExtension = "jpg"
        imagesLocator = os.path.join(imagesDirectoryPath, "*." + fileExtension)
        imagePaths = glob.glob(imagesLocator, recursive=True)
        imageInfos = []

        for imagePath in imagePaths:
            imageInfo = ImageInfo.getInstanceForImagePath(imagePath)
            imageInfos.append(imageInfo)

        return imageInfos

    @staticmethod
    def getInstanceForImagePath(imagePath: str):
        imageNumber = ImageInfo.__determineImageNumber(imagePath)
        return ImageInfo.getInstance(imageNumber, imagePath, None)

    @staticmethod
    def getInstance(imageNumber: int, imagePath: str, cropBox: CropBox):
        return ImageInfo(imageNumber, imagePath, cropBox)

    def __init__(self, imageNumber: int, imagePath: str, cropBox: CropBox):
        self.__imageNumber = imageNumber
        self.__imagePath = imagePath
        self.__cropBox = cropBox
        # Just using for dimension info, then discarding to preserve memory
        pilImage = self.getPilImage()
        self.__width = pilImage.width
        self.__height = pilImage.height

    def getImageNumber(self) -> int:
        return self.__imageNumber

    # lazy loading, to prevent huge amounts of memory being used
    def getPilImage(self) -> Image:
        originalPillImage = ImageInfo.__loadPILImageFromPath(self.__imagePath)

        if self.__cropBox is None:
            return originalPillImage

        return ImageInfo.__getPilImagePortion(originalPillImage, self.__cropBox)

    def getImagePath(self) -> str:
        return self.__imagePath

    def getWidth(self) -> int:
        return self.__width

    def getHeight(self) -> int:
        return self.__height

    @staticmethod
    def __determineImageNumber(imagePath) -> int:
        return os.path.split(imagePath)[-1][0:-4]

    @staticmethod
    def __loadPILImageFromPath(imagePath: str) -> Image:
        return image_processing.load_img(imagePath)

    @staticmethod
    def __getPilImagePortion(sourcePilImage: Image, cropBox: CropBox) -> Image:
        return sourcePilImage.crop((cropBox.getBeginX(), cropBox.getBeginY(), cropBox.getEndX(), cropBox.getEndY()))
