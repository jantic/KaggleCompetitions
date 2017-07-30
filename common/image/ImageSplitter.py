from common.image.CropBox import CropBox
from common.image.ImageInfo import ImageInfo


class ImageSplitter:
    @staticmethod
    def getImageDividedIntoVerticalHalves(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        fullHeight = sourceImageInfo.getHeight()
        halfWidth = ImageSplitter.__getHalfWidth(sourceImageInfo)

        # Left Half
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, 0, 0, halfWidth, fullHeight))
        # Right Half
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, halfWidth + 1, 0, halfWidth, fullHeight))
        return newImageInfos

    @staticmethod
    def getImageDividedIntoHorizontalHalves(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        fullWidth = sourceImageInfo.getWidth()
        halfHeight = ImageSplitter.__getHalfHeight(sourceImageInfo)

        # Top Half
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, 0, 0, fullWidth, halfHeight))
        # Bottom Half
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, 0, halfHeight + 1, fullWidth, halfHeight))
        return newImageInfos

    @staticmethod
    def getImageDividedIntoSquareQuadrants(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        halfHeight = ImageSplitter.__getHalfHeight(sourceImageInfo)
        halfWidth = ImageSplitter.__getHalfWidth(sourceImageInfo)

        # Top Left
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, 0, 0, halfWidth, halfHeight))
        # Top Right
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, halfWidth + 1, 0, halfWidth, halfHeight))
        # Bottom Left
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, 0, halfHeight + 1, halfWidth, halfHeight))
        # Bottom Right
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, halfWidth + 1, halfHeight + 1, halfWidth, halfHeight))
        return newImageInfos

    @staticmethod
    def getImageHalfCenter(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        halfHeight = ImageSplitter.__getHalfHeight(sourceImageInfo)
        halfWidth = ImageSplitter.__getHalfWidth(sourceImageInfo)
        quarterHeight = int(halfHeight / 2)
        quarterWidth = int(halfWidth / 2)

        # Top Center
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, quarterWidth, quarterHeight, halfWidth, halfHeight))
        return newImageInfos

    @staticmethod
    def getImageDividedIntoCrossQuadrants(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        halfHeight = ImageSplitter.__getHalfHeight(sourceImageInfo)
        halfWidth = ImageSplitter.__getHalfWidth(sourceImageInfo)
        quarterHeight = int(halfHeight / 2)
        quarterWidth = int(halfWidth / 2)

        # Top Center
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, quarterWidth, 0, halfWidth, halfHeight))
        # Bottom Center
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, quarterWidth, halfHeight + 1, halfWidth, halfHeight))
        # Left Center
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, 0, quarterHeight, halfWidth, halfHeight))
        # Right Center
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, halfWidth + 1, quarterHeight, halfWidth, halfHeight))
        return newImageInfos

    @staticmethod
    def getImageDividedIntoSquareThreeQuartersCorners(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []

        threeQuartersHeight = ImageSplitter.__getThreeQuartersHeight(sourceImageInfo)
        threeQuartersWidth = ImageSplitter.__getThreeQuartersWidth(sourceImageInfo)
        oneQuarterHeight = sourceImageInfo.getHeight() - threeQuartersHeight
        oneQuarterWidth = sourceImageInfo.getWidth() - threeQuartersWidth

        # Top Left
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, 0, 0, threeQuartersWidth, threeQuartersHeight))
        # Top Right
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, oneQuarterWidth + 1, 0, threeQuartersWidth, threeQuartersHeight))
        # Bottom Left
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, 0, oneQuarterHeight + 1, threeQuartersWidth, threeQuartersHeight))
        # Bottom Right
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, oneQuarterWidth + 1, oneQuarterHeight + 1, threeQuartersWidth, threeQuartersHeight))
        return newImageInfos

    @staticmethod
    def getImageDividedIntoThreeQuartersCross(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        threeQuartersHeight = ImageSplitter.__getThreeQuartersHeight(sourceImageInfo)
        threeQuartersWidth = ImageSplitter.__getThreeQuartersWidth(sourceImageInfo)
        oneQuarterHeight = sourceImageInfo.getHeight() - threeQuartersHeight
        oneQuarterWidth = sourceImageInfo.getWidth() - threeQuartersWidth

        # Top Center
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, oneQuarterWidth, 0, threeQuartersWidth, threeQuartersHeight))
        # Bottom Center
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, oneQuarterWidth, oneQuarterHeight + 1, threeQuartersWidth, threeQuartersHeight))
        # Left Center
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, 0, oneQuarterWidth, threeQuartersWidth, threeQuartersHeight))
        # Right Center
        newImageInfos.append(ImageSplitter.__getImagePortion(sourceImageInfo, oneQuarterHeight + 1, oneQuarterHeight, threeQuartersWidth, threeQuartersHeight))
        return newImageInfos

    @staticmethod
    def __getImagePortion(sourceImageInfo, beginX: int, beginY: int, width: int, height: int):
        cropBox = CropBox(beginX, beginY, width, height)
        return ImageInfo.getInstance(sourceImageInfo.getImageNumber(), sourceImageInfo.getImagePath(), cropBox)

    @staticmethod
    def __getHalfWidth(sourceImageInfo: ImageInfo) -> int:
        fullWidth = sourceImageInfo.getWidth()
        halfWidth = round(fullWidth / 2, 0) - 1
        return int(halfWidth)

    @staticmethod
    def __getHalfHeight(sourceImageInfo: ImageInfo) -> int:
        fullHeight = sourceImageInfo.getHeight()
        halfHeight = round(fullHeight / 2, 0) - 1
        return int(halfHeight)

    @staticmethod
    def __getThreeQuartersHeight(sourceImageInfo: ImageInfo) -> int:
        fullHeight = sourceImageInfo.getHeight()
        threeQuartersHeight = round(fullHeight * 3 / 4, 0) - 1
        return int(threeQuartersHeight)

    @staticmethod
    def __getThreeQuartersWidth(sourceImageInfo: ImageInfo) -> int:
        fullWidth = sourceImageInfo.getWidth()
        threeQuartersWidth = round(fullWidth * 3 / 4, 0) - 1
        return int(threeQuartersWidth)
