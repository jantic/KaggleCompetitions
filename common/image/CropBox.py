class CropBox:
    def __init__(self, begin_x: int, begin_y: int, width: int, height: int):
        self.__begin_x = begin_x
        self.__begin_y = begin_y
        self.__width = width
        self.__height = height

    def get_begin_x(self):
        return self.__begin_x

    def get_begin_y(self):
        return self.__begin_y

    def get_end_x(self):
        return self.__begin_x + self.__width

    def get_end_y(self):
        return self.__begin_y + self.__height

    def get_width(self):
        return self.__width

    def get_height(self):
        return self.__height
