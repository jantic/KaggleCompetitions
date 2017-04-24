from common.image.ImageInfo import ImageInfo


class ImageSplitter:


    @staticmethod
    def getImageDividedIntoVerticalHalves(sourceImageInfo : ImageInfo) -> [ImageInfo] :
        newImageInfos = []
        fullHeight = sourceImageInfo.getHeight()
        halfWidth = ImageSplitter.getHalfWidth(sourceImageInfo)

        #Left Half
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, 0, halfWidth, fullHeight))
        #Right Half
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, halfWidth+1, 0, halfWidth, fullHeight))
        return newImageInfos

    @staticmethod
    def getImageDividedIntoHorizontalHalves(sourceImageInfo : ImageInfo) -> [ImageInfo] :
        newImageInfos = []
        fullWidth= sourceImageInfo.getWidth()
        halfHeight = ImageSplitter.getHalfHeight(sourceImageInfo)

        #Top Half
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, 0, fullWidth, halfHeight))
        #Bottom Half
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, halfHeight+1, fullWidth, halfHeight))
        return newImageInfos

    @staticmethod
    def getImageDividedIntoQuadrants(sourceImageInfo : ImageInfo) -> [ImageInfo] :
        newImageInfos = []
        halfHeight = ImageSplitter.getHalfHeight(sourceImageInfo)
        halfWidth = ImageSplitter.getHalfWidth(sourceImageInfo)

        #Top Left
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, 0, halfWidth, halfHeight))
        #Top Right
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, halfWidth+1, 0, halfWidth, halfHeight))
        #Bottom Left
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, 0, halfHeight+1, halfWidth, halfHeight))
        #Bottom Right
        newImageInfos.append(ImageSplitter.getImagePortion(sourceImageInfo, halfWidth+1, halfHeight+1, halfWidth, halfHeight))
        return newImageInfos

    @staticmethod
    def getImagePortion(sourceImageInfo : ImageInfo, beginX : int, beginY : int, width : int, height : int) -> ImageInfo :
        sourcePilImage= sourceImageInfo.getPilImage()
        endX = beginX + width
        endY = beginY + height
        newPilImage = sourcePilImage.crop((beginX, beginY, endX, endY))
        imageNumber = sourceImageInfo.getImageNumber()
        return ImageInfo.getInstance(width, height, imageNumber, newPilImage)

    @staticmethod
    def getHalfWidth(sourceImageInfo : ImageInfo) -> int:
        fullWidth = sourceImageInfo.getWidth()
        halfWidth = round(fullWidth / 2, 0)-1
        return int(halfWidth)

    @staticmethod
    def getHalfHeight(sourceImageInfo : ImageInfo) -> int:
        fullHeight = sourceImageInfo.getHeight()
        halfHeight= round(fullHeight / 2, 0)-1
        return int(halfHeight)

