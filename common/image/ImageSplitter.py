from common.image.CropBox import CropBox
from common.image.ImageInfo import ImageInfo


class ImageSplitter:
    @staticmethod
    def get_image_divided_into_vertical_halves(source_image_info: ImageInfo) -> [ImageInfo]:
        new_image_infos = []
        full_height = source_image_info.get_height()
        half_width = ImageSplitter.__get_half_width(source_image_info)

        # Left Half
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, 0, 0, half_width, full_height))
        # Right Half
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, half_width + 1, 0, half_width, full_height))
        return new_image_infos

    @staticmethod
    def get_image_divided_into_horizontal_halves(source_image_info: ImageInfo) -> [ImageInfo]:
        new_image_infos = []
        full_width = source_image_info.get_width()
        half_height = ImageSplitter.__get_half_height(source_image_info)

        # Top Half
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, 0, 0, full_width, half_height))
        # Bottom Half
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, 0, half_height + 1, full_width, half_height))
        return new_image_infos

    @staticmethod
    def get_image_divided_into_square_quadrants(source_image_info: ImageInfo) -> [ImageInfo]:
        new_image_infos = []
        half_height = ImageSplitter.__get_half_height(source_image_info)
        half_width = ImageSplitter.__get_half_width(source_image_info)

        # Top Left
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, 0, 0, half_width, half_height))
        # Top Right
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, half_width + 1, 0, half_width, half_height))
        # Bottom Left
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, 0, half_height + 1, half_width, half_height))
        # Bottom Right
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, half_width + 1, half_height + 1, half_width, half_height))
        return new_image_infos

    @staticmethod
    def get_image_half_center(source_image_info: ImageInfo) -> [ImageInfo]:
        new_image_infos = []
        half_height = ImageSplitter.__get_half_height(source_image_info)
        half_width = ImageSplitter.__get_half_width(source_image_info)
        quarter_height = int(half_height / 2)
        quarter_width = int(half_width / 2)

        # Top Center
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, quarter_width, quarter_height, half_width, half_height))
        return new_image_infos

    @staticmethod
    def get_image_divided_into_cross_quadrants(source_image_info: ImageInfo) -> [ImageInfo]:
        new_image_infos = []
        half_height = ImageSplitter.__get_half_height(source_image_info)
        half_width = ImageSplitter.__get_half_width(source_image_info)
        quarter_height = int(half_height / 2)
        quarter_width = int(half_width / 2)

        # Top Center
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, quarter_width, 0, half_width, half_height))
        # Bottom Center
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, quarter_width, half_height + 1, half_width, half_height))
        # Left Center
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, 0, quarter_height, half_width, half_height))
        # Right Center
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, half_width + 1, quarter_height, half_width, half_height))
        return new_image_infos

    @staticmethod
    def get_image_divided_into_square_three_quarters_corners(source_image_info: ImageInfo) -> [ImageInfo]:
        new_image_infos = []

        three_quarters_height = ImageSplitter.__get_three_quarters_height(source_image_info)
        three_quarters_width = ImageSplitter.__get_three_quarters_width(source_image_info)
        one_quarter_height = source_image_info.get_height() - three_quarters_height
        one_quarter_width = source_image_info.get_width() - three_quarters_width

        # Top Left
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, 0, 0, three_quarters_width, three_quarters_height))
        # Top Right
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, one_quarter_width + 1, 0, three_quarters_width, three_quarters_height))
        # Bottom Left
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, 0, one_quarter_height + 1, three_quarters_width, three_quarters_height))
        # Bottom Right
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, one_quarter_width + 1, one_quarter_height + 1, three_quarters_width, three_quarters_height))
        return new_image_infos

    @staticmethod
    def get_image_divided_into_three_quarters_cross(source_image_info: ImageInfo) -> [ImageInfo]:
        new_image_infos = []
        three_quarters_height = ImageSplitter.__get_three_quarters_height(source_image_info)
        three_quarters_width = ImageSplitter.__get_three_quarters_width(source_image_info)
        one_quarter_height = source_image_info.get_height() - three_quarters_height
        one_quarter_width = source_image_info.get_width() - three_quarters_width

        # Top Center
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, one_quarter_width, 0, three_quarters_width, three_quarters_height))
        # Bottom Center
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, one_quarter_width, one_quarter_height + 1, three_quarters_width, three_quarters_height))
        # Left Center
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, 0, one_quarter_width, three_quarters_width, three_quarters_height))
        # Right Center
        new_image_infos.append(ImageSplitter.__get_image_portion(source_image_info, one_quarter_height + 1, one_quarter_height, three_quarters_width, three_quarters_height))
        return new_image_infos

    @staticmethod
    def __get_image_portion(source_image_info, begin_x: int, begin_y: int, width: int, height: int):
        crop_box = CropBox(begin_x, begin_y, width, height)
        return ImageInfo.get_instance(source_image_info.get_image_number(), source_image_info.get_image_path(), crop_box)

    @staticmethod
    def __get_half_width(source_image_info: ImageInfo) -> int:
        full_width = source_image_info.get_width()
        half_width = round(full_width / 2, 0) - 1
        return int(half_width)

    @staticmethod
    def __get_half_height(source_image_info: ImageInfo) -> int:
        full_height = source_image_info.get_height()
        half_height = round(full_height / 2, 0) - 1
        return int(half_height)

    @staticmethod
    def __get_three_quarters_height(source_image_info: ImageInfo) -> int:
        full_height = source_image_info.get_height()
        three_quarters_height = round(full_height * 3 / 4, 0) - 1
        return int(three_quarters_height)

    @staticmethod
    def __get_three_quarters_width(source_image_info: ImageInfo) -> int:
        full_width = source_image_info.get_width()
        three_quarters_width = round(full_width * 3 / 4, 0) - 1
        return int(three_quarters_width)
