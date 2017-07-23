from common.image.CropBox import CropBox
from common.image.ImageInfo import ImageInfo


class ImageSplitter:
    @staticmethod
    def getImageDividedIntoVerticalHalves(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        fullHeight = sourceImageInfo.getHeight()
        halfWidth = ImageSplitter.getHalfWidth(sourceImageInfo)

        # Left Half
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, 0, halfWidth, fullHeight))
        # Right Half
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, halfWidth + 1, 0, halfWidth, fullHeight))
        return newImageInfos

    @staticmethod
    def getImageDividedIntoHorizontalHalves(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        fullWidth = sourceImageInfo.getWidth()
        halfHeight = ImageSplitter.getHalfHeight(sourceImageInfo)

        # Top Half
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, 0, fullWidth, halfHeight))
        # Bottom Half
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, halfHeight + 1, fullWidth, halfHeight))
        return newImageInfos

    @staticmethod
    def getImageDividedIntoSquareQuadrants(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        halfHeight = ImageSplitter.getHalfHeight(sourceImageInfo)
        halfWidth = ImageSplitter.getHalfWidth(sourceImageInfo)

        # Top Left
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, 0, halfWidth, halfHeight))
        # Top Right
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, halfWidth + 1, 0, halfWidth, halfHeight))
        # Bottom Left
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, halfHeight + 1, halfWidth, halfHeight))
        # Bottom Right
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, halfWidth + 1, halfHeight + 1, halfWidth, halfHeight))
        return newImageInfos

    @staticmethod
    def getImageHalfCenter(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        halfHeight = ImageSplitter.getHalfHeight(sourceImageInfo)
        halfWidth = ImageSplitter.getHalfWidth(sourceImageInfo)
        quarterHeight = int(halfHeight / 2)
        quarterWidth = int(halfWidth / 2)

        # Top Center
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, quarterWidth, quarterHeight, halfWidth, halfHeight))
        return newImageInfos

    @staticmethod
    def getImageDividedIntoCrossQuadrants(sourceImageInfo: ImageInfo) -> [ImageInfo]:
        newImageInfos = []
        halfHeight = ImageSplitter.getHalfHeight(sourceImageInfo)
        halfWidth = ImageSplitter.getHalfWidth(sourceImageInfo)
        quarterHeight = int(halfHeight / 2)
        quarterWidth = int(halfWidth / 2)

        # Top Center
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, quarterWidth, 0, halfWidth, halfHeight))
        # Bottom Center
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, quarterWidth, halfHeight + 1, halfWidth, halfHeight))
        # Left Center
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, quarterHeight, halfWidth, halfHeight))
        # Right Center
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, halfWidth + 1, quarterHeight, halfWidth, halfHeight))
        return newImageInfos

    @staticmethod
    def getImagePortion(sourceImageInfo, beginX: int, beginY: int, width: int, height: int):
        cropBox = CropBox(beginX, beginY, width, height)
        return ImageInfo.getInstance(sourceImageInfo.getImageNumber(), sourceImageInfo.getImagePath(), cropBox)

    @staticmethod
    def getHalfWidth(sourceImageInfo: ImageInfo) -> int:
        fullWidth = sourceImageInfo.getWidth()
        halfWidth = round(fullWidth / 2, 0) - 1
        return int(halfWidth)

    @staticmethod
    def getHalfHeight(sourceImageInfo: ImageInfo) -> int:
        fullHeight = sourceImageInfo.getHeight()
        halfHeight = round(fullHeight / 2, 0) - 1
        return int(halfHeight)
