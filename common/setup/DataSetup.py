import shutil
from glob import glob
from numpy.random import permutation
import os

from common.setup.ImagesDirectoryCreationMode import ImagesDirectoryCreationMode
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import concurrent.futures

class DataSetup:
    def establish_working_data_directory_if_needed(self, source_directory: str, destination_directory: str,
                destination_sample_directory: str, image_file_extension='jpg', valid_to_test_ratio=0.1, sample_ratio=0.02,
                train_augment_factor=0):
        source_directory=DataSetup._cleanup_directory_path(source_directory)
        destination_directory=DataSetup._cleanup_directory_path(destination_directory)
        DataSetup._establish_directory_if_needed(destination_directory)

        if not self.__need_to_establish_working_data_directory(destination_directory):
            return

        destination_training_data_directory = destination_directory + '/train/'
        destination_validation_data_directory = destination_directory + '/valid/'

        self.__establish_training_and_test_data(source_directory=source_directory, destination_directory=destination_directory,
                                        image_file_extension=image_file_extension)

        self._establish_validation_data(training_directory=destination_training_data_directory, valid_directory=destination_validation_data_directory,
                                        image_file_extension=image_file_extension, valid_to_test_ratio=valid_to_test_ratio)

        self._establish_sample_data(main_data_directory=destination_directory, sample_directory=destination_sample_directory,
                                    image_file_extension=image_file_extension, sample_ratio=sample_ratio)

        self._augment_training_data_if_applicable(training_directory=destination_training_data_directory,
                                                  train_augment_factor=train_augment_factor, image_file_extension=image_file_extension)

    def _augment_training_data_if_applicable(self, training_directory: str, train_augment_factor: int, image_file_extension: str):
        if train_augment_factor <= 0:
            return

        sub_directory_paths = self._get_sub_directories(training_directory)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for sub_directory_path in sub_directory_paths:
                source_images = DataSetup._get_files_with_extension(sub_directory_path, image_file_extension)
                futures = []
                for source_image in source_images:
                    futures.append(executor.submit(self._add_augmented_images, train_augment_factor, source_image))

                concurrent.futures.wait(futures)


    def _establish_training_and_test_data(self, source_directory: str, destination_directory: str, image_file_extension: str):
        source_directory=DataSetup._cleanup_directory_path(source_directory)
        destination_directory=DataSetup._cleanup_directory_path(destination_directory)
        DataSetup._establish_directory_if_needed(destination_directory)
        source_data_directoryPaths = DataSetup._get_sub_directories(source_directory)

        for source_data_directory_path in source_data_directoryPaths:
            new_directory_path = str.replace(source_data_directory_path, source_directory, destination_directory)
            DataSetup._establish_directory_if_needed(new_directory_path)
            self._create_new_images_directory(source_data_directory_path, new_directory_path,
                                              image_file_extension, 1.0, ImagesDirectoryCreationMode.COPY)

    def _need_to_establish_working_data_directory(self, destination_directory):
        destination_sub_directory_names = DataSetup._get_sub_directories(destination_directory)
        return len(destination_sub_directory_names) == 0

    def _establish_validation_data(self, training_directory: str, valid_directory: str, image_file_extension: str, valid_to_test_ratio: float):
        training_directory=DataSetup._cleanup_directory_path(training_directory)
        valid_directory=DataSetup._cleanup_directory_path(valid_directory)
        DataSetup._establish_directory_if_needed(valid_directory)
        self._create_new_images_directory(training_directory, valid_directory, image_file_extension, valid_to_test_ratio, ImagesDirectoryCreationMode.MOVE)

    def _establish_sample_data(self, main_data_directory: str, sample_directory: str, image_file_extension: str, sample_ratio: float):
        main_data_directory=DataSetup._cleanup_directory_path(main_data_directory)
        sample_directory=DataSetup._cleanup_directory_path(sample_directory)
        DataSetup._establish_directory_if_needed(sample_directory)
        main_data_directoryPaths = DataSetup._get_sub_directories(main_data_directory)

        for main_data_directory_path in main_data_directoryPaths:
            new_sample_directory_path = str.replace(main_data_directory_path, main_data_directory, sample_directory)
            DataSetup._establish_directory_if_needed(new_sample_directory_path)
            self._create_new_images_directory(main_data_directory_path, new_sample_directory_path, image_file_extension, sample_ratio,
                                              ImagesDirectoryCreationMode.COPY)

    def _create_new_images_directory(self, source_dir: str, destination_dir: str, image_file_extension: str, ratio_to_copy: float,
                                     creation_mode: ImagesDirectoryCreationMode):
        source_sub_directories = DataSetup._get_sub_directories(source_dir)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for source_sub_directory in source_sub_directories:
                self.__create_new_images_subdirectory(source_sub_directory, source_dir, destination_dir, image_file_extension, ratio_to_copy, creation_mode)
                futures.append(executor.submit(DataSetup.__create_new_images_subdirectory, source_sub_directory,
                                               source_dir, destination_dir, image_file_extension, ratio_to_copy, creation_mode))
            concurrent.futures.wait(futures)

    @staticmethod
    def __create_new_images_subdirectory(source_sub_directory: str, source_dir: str, destination_dir: str, image_file_extension: str, ratio_to_copy: float,
                                         creation_mode: ImagesDirectoryCreationMode):
        destination_sub_directory = str.replace(source_sub_directory, source_dir, destination_dir)
        DataSetup._establish_directory_if_needed(destination_sub_directory)
        source_images = DataSetup._get_files_with_extension(source_sub_directory, image_file_extension)
        num_images_to_copy = int(round(ratio_to_copy * len(source_images), 0))
        images_to_move_or_copy = permutation(source_images)[:num_images_to_copy+1]

        for source_image_to_move_or_copy in images_to_move_or_copy:
            new_image_path = str.replace(source_image_to_move_or_copy, source_dir, destination_dir)
            if creation_mode == ImagesDirectoryCreationMode.MOVE:
                shutil.move(source_image_to_move_or_copy, new_image_path)
            else:
                shutil.copy(source_image_to_move_or_copy, new_image_path)

    def _add_augmented_images(self, augment_factor: int, original_image_path: str):
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
    def _cleanup_directory_path(directory: str):
        return str.replace(directory, "/", "\\")

    @staticmethod
    def _get_sub_directories(directory: str):
        return glob(directory + "/*/")

    @staticmethod
    def _get_files_with_extension(directory: str, extension: str):
        return glob(directory + "/*." + extension)

    @staticmethod
    def _establish_directory_if_needed(directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

