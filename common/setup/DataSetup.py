import shutil
from glob import glob
from numpy.random import permutation
import os

from common.setup.ImagesDirectoryCreationMode import ImagesDirectoryCreationMode
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class DataSetup:
    @staticmethod
    def establish_working_data_directory_if_needed(source_directory: str, destination_directory: str,
                destination_sample_directory: str, image_file_extension='jpg', valid_to_test_ratio=0.1, sample_ratio=0.02,
                train_augment_factor=0):
        source_training_directory=DataSetup.__cleanup_directory_path(source_directory)
        destination_directory=DataSetup.__cleanup_directory_path(destination_directory)
        DataSetup.__establish_directory_if_needed(destination_directory)

        if not DataSetup.__need_to_establish_working_data_directory(destination_directory):
            return

        destination_training_data_directory = destination_directory + '/test/'
        destination_validation_data_directory = destination_directory + '/valid/'

        DataSetup.__establish_training_and_test_data(source_directory=source_directory, destination_directory=destination_directory,
                                        image_file_extension=image_file_extension, train_augment_factor=train_augment_factor)

        DataSetup.__establish_validation_data(training_directory=destination_training_data_directory,valid_directory=destination_validation_data_directory,
                                              image_file_extension=image_file_extension, valid_to_test_ratio=valid_to_test_ratio)

        DataSetup.__establish_sample_data(main_data_directory=destination_directory, sample_directory=destination_sample_directory,
                                          image_file_extension=image_file_extension, sample_ratio=sample_ratio)

    @staticmethod
    def __establish_training_and_test_data(source_directory: str, destination_directory: str, image_file_extension: str, train_augment_factor: int):
        source_directory=DataSetup.__cleanup_directory_path(source_directory)
        destination_directory=DataSetup.__cleanup_directory_path(destination_directory)
        DataSetup.__establish_directory_if_needed(destination_directory)
        source_data_directoryPaths = DataSetup.__get_sub_directories(source_directory)
        test_augment_factor = 0

        for source_data_directory_path in source_data_directoryPaths:
            new_directory_path = str.replace(source_data_directory_path, source_directory, destination_directory)
            DataSetup.__establish_directory_if_needed(new_directory_path)
            last_directory_name = os.path.basename(os.path.normpath(new_directory_path))
            is_test_directory = last_directory_name.startswith('test')
            augment_factor = 0 if is_test_directory else train_augment_factor
            DataSetup.__create_new_images_directory(source_data_directory_path, new_directory_path,
                    image_file_extension, 1.0, ImagesDirectoryCreationMode.COPY, augment_factor=augment_factor)


    @staticmethod
    def __need_to_establish_working_data_directory(destination_directory):
        destination_sub_directory_names = DataSetup.__get_sub_directories(destination_directory)
        return len(destination_sub_directory_names) == 0

    @staticmethod
    def __establish_validation_data(training_directory: str, valid_directory: str, image_file_extension: str, valid_to_test_ratio: float):
        training_directory=DataSetup.__cleanup_directory_path(training_directory)
        valid_directory=DataSetup.__cleanup_directory_path(valid_directory)
        DataSetup.__establish_directory_if_needed(valid_directory)
        DataSetup.__create_new_images_directory(training_directory, valid_directory, image_file_extension, valid_to_test_ratio, ImagesDirectoryCreationMode.MOVE)


    @staticmethod
    def __establish_sample_data(main_data_directory: str, sample_directory: str, image_file_extension: str, sample_ratio: float):
        main_data_directory=DataSetup.__cleanup_directory_path(main_data_directory)
        sample_directory=DataSetup.__cleanup_directory_path(sample_directory)
        DataSetup.__establish_directory_if_needed(sample_directory)
        main_data_directoryPaths = DataSetup.__get_sub_directories(main_data_directory)

        for main_data_directory_path in main_data_directoryPaths:
            new_sample_directory_path = str.replace(main_data_directory_path, main_data_directory, sample_directory)
            DataSetup.__establish_directory_if_needed(new_sample_directory_path)
            DataSetup.__create_new_images_directory(main_data_directory_path, new_sample_directory_path, image_file_extension, sample_ratio,
                                                    ImagesDirectoryCreationMode.COPY)

    @staticmethod
    def __create_new_images_directory(source_dir: str, destination_dir: str, image_file_extension: str, ratio_to_copy: float,
                                      creation_mode: ImagesDirectoryCreationMode, augment_factor=0):
        source_sub_directories = DataSetup.__get_sub_directories(source_dir)

        for source_sub_directory in source_sub_directories:
            destination_sub_directory = str.replace(source_sub_directory, source_dir, destination_dir)
            DataSetup.__establish_directory_if_needed(destination_sub_directory)
            source_images = DataSetup.__get_files_with_extension(source_sub_directory, image_file_extension)
            num_images_to_copy = int(round(ratio_to_copy * len(source_images), 0))
            images_to_move_or_copy = permutation(source_images)[:num_images_to_copy]

            for source_image_to_move_or_copy in images_to_move_or_copy:
                new_image_path = str.replace(source_image_to_move_or_copy, source_dir, destination_dir)
                if creation_mode == ImagesDirectoryCreationMode.MOVE:
                    shutil.move(source_image_to_move_or_copy, new_image_path)
                else:
                    shutil.copy(source_image_to_move_or_copy, new_image_path)
                    if augment_factor > 0:
                        DataSetup.__add_augmented_images(new_image_path, augment_factor)

    @staticmethod
    def __add_augmented_images(original_image_path: str, augment_factor: int):
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            channel_shift_range=0.2,
            fill_mode='nearest')

        original_image = load_img(original_image_path)
        x = img_to_array(original_image)
        x = x.reshape((1,) + x.shape)

        save_to_dir = os.path.dirname(os.path.abspath(original_image_path))
        save_prefix = os.path.splitext(os.path.basename(original_image_path))[0]+'_aug'

        image_num = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_to_dir, save_prefix=save_prefix, save_format='jpeg'):
            image_num += 1
            if image_num >= augment_factor:
                break


    @staticmethod
    def __cleanup_directory_path(directory: str):
        return str.replace(directory, "/", "\\")

    @staticmethod
    def __get_sub_directories(directory: str):
        return glob(directory + "/*/")

    @staticmethod
    def __get_files_with_extension(directory: str, extension: str):
        return glob(directory + "/*." + extension)

    @staticmethod
    def __establish_directory_if_needed(directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

