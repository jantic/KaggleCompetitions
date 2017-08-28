import shutil
from glob import glob
from numpy.random import permutation
import os


class DataSetup:
    @staticmethod
    def establish_validation_data_if_needed(training_directory: str, valid_directory: str, image_file_extension='jpg', valid_to_test_ratio=0.1):
        training_directory = str.replace(training_directory, "/", "\\")
        valid_directory = str.replace(valid_directory, "/", "\\")

        if not os.path.exists(valid_directory):
            os.mkdir(valid_directory)

        validation_class_directory_names = glob(valid_directory + "/*/")

        if len(validation_class_directory_names) > 0:
            return

        training_class_directory_paths = glob(training_directory + "/*/")

        for training_class_directory_path in training_class_directory_paths:
            new_valid_class_directory_path = str.replace(training_class_directory_path, training_directory, valid_directory)

            if not os.path.exists(new_valid_class_directory_path):
                os.mkdir(new_valid_class_directory_path)

            training_images = glob(training_class_directory_path + "/*." + image_file_extension)
            num_valid_images = int(round(valid_to_test_ratio * len(training_images), 0))
            test_images_to_move = permutation(training_images)[:num_valid_images]

            for test_image_to_move in test_images_to_move:
                new_valid_image_path = str.replace(test_image_to_move, training_directory, valid_directory)
                shutil.move(test_image_to_move, new_valid_image_path)

    @staticmethod
    def establish_sample_data_if_needed(main_data_directory: str, sample_directory: str, image_file_extension='jpg', sample_ratio=0.02):
        main_data_directory = str.replace(main_data_directory, "/", "\\")
        sample_directory = str.replace(sample_directory, "/", "\\")

        if not os.path.exists(sample_directory):
            os.mkdir(sample_directory)

        sample_directoryNames = glob(sample_directory + "/*/")

        if len(sample_directoryNames) > 0:
            return

        main_data_directoryPaths = glob(main_data_directory + "/*/")

        for main_data_directoryPath in main_data_directoryPaths:
            new_sample_directory_path = str.replace(main_data_directoryPath, main_data_directory, sample_directory)

            if not os.path.exists(new_sample_directory_path):
                os.mkdir(new_sample_directory_path)

            main_sub_directory_paths = glob(main_data_directoryPath + "/*/")

            for main_sub_directory_path in main_sub_directory_paths:
                new_sample_sub_directory_path = str.replace(main_sub_directory_path, main_data_directory, sample_directory)

                if not os.path.exists(new_sample_sub_directory_path):
                    os.mkdir(new_sample_sub_directory_path)

                source_images = glob(main_sub_directory_path + "/*." + image_file_extension)
                num_sample_images = int(round(sample_ratio * len(source_images), 0))
                source_images_to_copy = permutation(source_images)[:num_sample_images]

                for source_image_to_copy in source_images_to_copy:
                    new_sample_image_path = str.replace(source_image_to_copy, main_data_directory, sample_directory)
                    shutil.copy(source_image_to_copy, new_sample_image_path)


