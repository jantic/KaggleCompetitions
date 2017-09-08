from __future__ import division, print_function

from collections import OrderedDict
from importlib import reload

import numpy as np
import pandas as pd
import pathlib2
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary
from common.model.deeplearning.imagerec.MasterImageClassifier import MasterImageClassifier
from common.model.deeplearning.imagerec.pretrained import vgg16
from common.model.deeplearning.imagerec.pretrained.vgg16 import Vgg16
from common.setup.DataSetup import DataSetup
from common.utils import utils
from common.visualization.ImagePerformanceVisualizer import ImagePerformanceVisualizer
import cProfile


def write_predictions_to_csv(pred_summaries: [PredictionsSummary]):
    if len(pred_summaries) == 0:
        return

    records = []

    for pred_summary in pred_summaries:
        image_name = pathlib2.PureWindowsPath(pred_summary.get_image_info().get_image_path()).name

        predictions = pred_summary.get_all_predictions()
        raw_confidence_array = np.zeros(10)

        for prediction in predictions:
            confidence = prediction.get_confidence()
            raw_confidence_array[prediction.get_class_id()] = confidence

        clipped_confidence = utils.do_clip(raw_confidence_array, 0.99)

        row = OrderedDict([('img', image_name), ('c0', clipped_confidence[0]), ('c1', clipped_confidence[1]), ('c2', clipped_confidence[2]),
                           ('c3', clipped_confidence[3]), ('c4', clipped_confidence[4]), ('c5', clipped_confidence[5]), ('c6', clipped_confidence[6]),
                           ('c7', clipped_confidence[7]), ('c8', clipped_confidence[8]), ('c9', clipped_confidence[9])])

        records.append(row)

    df = pd.DataFrame.from_records(records)
    df = df.sort_values('img')
    df.to_csv('submission.csv', index=False)

run_main_test = False
refine_training = True
image_splitting = False
visualize_performance = True
visualization_class = 'c0'
use_sample = False
number_of_epochs = 6
training_batch_size = 64
validation_batch_size = 64
test_batch_size = 64

reload(utils)
np.set_printoptions(precision=4, linewidth=100)
reload(vgg16)

data_directory="data/"
source_directory= data_directory + "source/"

main_directory = data_directory + "main/"
main_training_set_path = main_directory + "train"
main_validation_set_path = main_directory + "valid"
main_test_set_path = main_directory + "test/"
main_cache_path = "./cache/main/"
main_steps_per_epoch = 500

sample_directory = data_directory + "sample/"
sample_training_set_path = sample_directory + "train"
sample_validation_set_path = sample_directory + "valid"
sample_test_set_path = sample_directory + "test/"
sample_cache_path = "./cache/sample/"
sample_steps_per_epoch = 10

DataSetup.establish_working_data_directory_if_needed(source_directory=source_directory, destination_directory=main_directory,
    destination_sample_directory=sample_directory, image_file_extension='jpg', valid_to_test_ratio=0.1, sample_ratio=0.04,train_augment_factor=10)

training_set_path = sample_training_set_path if use_sample else main_training_set_path
validation_set_path = sample_validation_set_path if use_sample else main_validation_set_path
test_set_path = sample_test_set_path if use_sample else main_test_set_path
cache_directory = sample_cache_path if use_sample else main_cache_path
steps_per_epoch = sample_steps_per_epoch if use_sample else main_steps_per_epoch

vgg = Vgg16(load_weights_from_cache=True, training_images_path=training_set_path, training_batch_size=training_batch_size, validation_images_path=validation_set_path,
            validation_batch_size=validation_batch_size, cache_directory=cache_directory, num_dense_layers_to_retrain=4, fast_conv_cache_training=True, drop_out=0.5)

if refine_training:
    vgg.refine_training(steps_per_epoch=steps_per_epoch, number_of_epochs=number_of_epochs)

image_classifier = MasterImageClassifier(vgg)

if run_main_test:
    prediction_summaries = image_classifier.get_all_predictions(test_set_path, False, test_batch_size)
    write_predictions_to_csv(prediction_summaries)

if visualize_performance:
    test_result_summaries = []
    classes = vgg.get_classes()

    for clazz in classes:
        visualization_test_path = validation_set_path + "/" + clazz
        test_result_summaries.extend(image_classifier.get_all_test_results(visualization_test_path, image_splitting, test_batch_size, clazz))

    ImagePerformanceVisualizer.do_visualizations(test_result_summaries, visualization_class, 5, True, True, True, True)

