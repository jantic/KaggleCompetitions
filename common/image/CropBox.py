class CropBox:
    def __init__(self, beginX: int, beginY: int, width: int, height: int):
        self.__beginX = beginX
        self.__beginY = beginY
        self.__width = width
        self.__height = height

    def getBeginX(self):
        return self.__beginX

    def getBeginY(self):
        return self.__beginY

    def getEndX(self):
        return self.__beginX + self.__width

    def getEndY(self):
        return self.__beginY + self.__height

    def getWidth(self):
        return self.__width

    def getHeight(self):
        return self.__height
