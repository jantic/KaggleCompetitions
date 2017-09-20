import math as math
import os

import pandas as pd
from common.setup.DataSetup import DataSetup
from common.setup.ImagesDirectoryCreationMode import ImagesDirectoryCreationMode
import shutil
from numpy.random import permutation


class DistractedDriverDataSetup(DataSetup):
    def __init__(self):
        super(DistractedDriverDataSetup, self).__init__()

    #Creating validation set with different drivers rather than randomly moved images, to reduce overfitting when running
    #validation tests
    def _establish_validation_data(self, training_directory: str, valid_directory: str, image_file_extension: str, valid_to_test_ratio: float):
        image_file_extension = 'jpg'
        image_to_driver_csv = pd.read_csv('driver_imgs_list.csv')
        all_drivers = image_to_driver_csv.subject.unique()
        num_drivers = len(all_drivers)
        num_drivers_validation = math.ceil(valid_to_test_ratio * num_drivers)
        validation_drivers = permutation(all_drivers)[:num_drivers_validation+1]
        image_to_driver = image_to_driver_csv.set_index('img')['subject'].to_dict()
        training_directory=DataSetup._cleanup_directory_path(training_directory)
        valid_directory=DataSetup._cleanup_directory_path(valid_directory)
        DataSetup._establish_directory_if_needed(valid_directory)
        source_sub_directories = DataSetup._get_sub_directories(training_directory)

        for source_sub_directory in source_sub_directories:
            destination_sub_directory = str.replace(source_sub_directory, training_directory, valid_directory)
            DataSetup._establish_directory_if_needed(destination_sub_directory)
            source_images = DataSetup._get_files_with_extension(source_sub_directory, image_file_extension)
            images_to_move_or_copy = [image for image in source_images if image_to_driver[os.path.basename(image)] in validation_drivers]

            for source_image_to_move_or_copy in images_to_move_or_copy:
                new_image_path = str.replace(source_image_to_move_or_copy, training_directory, valid_directory)
                shutil.move(source_image_to_move_or_copy, new_image_path)


