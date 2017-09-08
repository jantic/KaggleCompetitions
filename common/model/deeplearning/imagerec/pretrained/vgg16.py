from __future__ import absolute_import
from __future__ import division, print_function

import glob
import os.path
import re

import keras
import numpy as np
from keras import layers
from keras.layers import BatchNormalization
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, Iterator
from keras.utils.data_utils import get_file
from keras.models import load_model

from common.model.deeplearning.imagerec.optimization.ConvCacheIterator import ConvCacheIterator
from common.utils.utils import *
from common.model.deeplearning.imagerec.BatchImagePredictionRequestInfo import BatchImagePredictionRequestInfo
from common.model.deeplearning.imagerec.IImageRecModel import IImageRecModel
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from common.model.deeplearning.imagerec.ImagePredictionResult import ImagePredictionResult


class Vgg16(IImageRecModel):
    """The VGG 16 Imagenet model"""

    def __init__(self, load_weights_from_cache: bool, training_images_path: str, training_batch_size: int, validation_images_path: str,
                 validation_batch_size: int, cache_directory: str, num_dense_layers_to_retrain: int, fast_conv_cache_training=True,
                 drop_out=0.0):
        self.FAST_CONV_CACHE_TRAINING = fast_conv_cache_training
        self.TRAINING_BATCH_SIZE = training_batch_size
        self.VALIDATION_BATCH_SIZE = validation_batch_size
        self.TRAINING_BATCHES = self.__get_batches(training_images_path, shuffle=True, batch_size=training_batch_size)
        self.VALIDATION_BATCHES = self.__get_batches(validation_images_path, shuffle=True, batch_size=validation_batch_size)
        self.VGG_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
        self.ORIGINAL_MODEL_WEIGHTS_URL = 'http://files.fast.ai/models/'
        self.CACHE_DIRECTORY = cache_directory
        self.CONV_TRAIN_CACHE_PATH = self.CACHE_DIRECTORY + 'train_convlayer_features.bc'
        self.CONV_VALID_CACHE_PATH = self.CACHE_DIRECTORY + 'valid_convlayer_features.bc'
        self.LOAD_WEIGHTS_FROM_CACHE = load_weights_from_cache
        self.NUM_DENSE_LAYERS_TO_RETRAIN = num_dense_layers_to_retrain
        self.DROP_OUT = drop_out
        self.__initialize_model()

    def refine_training(self, steps_per_epoch: int, number_of_epochs: int):
        latest_saved_filename = self.__get_latest_saved_weights_file_name()
        latest_saved_epoch = self.__determine_epoch_num_from_weights_file_name(latest_saved_filename)
        initial_epoch = max(latest_saved_epoch, 0) if self.__can_load_weights_from_cache() else 0
        self.__fit(self.TRAINING_BATCHES, self.VALIDATION_BATCHES, steps_per_epoch=steps_per_epoch, nb_epoch=number_of_epochs, initial_epoch=initial_epoch)

    def predict(self, image_prediction_requests: [ImagePredictionRequest], batch_size: int, details=False) -> [ImagePredictionResult]:
        verbose = 1 if details else 0
        batch_request_info = BatchImagePredictionRequestInfo.get_instance(image_prediction_requests, self.get_image_width(), self.get_image_height())
        batch_confidences = self.model.predict(batch_request_info.get_image_array(), batch_size=batch_size, verbose=verbose)
        image_prediction_results = ImagePredictionResult.generate_image_prediction_results(batch_confidences, batch_request_info, self.classes)
        return image_prediction_results

    def get_image_width(self):
        return 224

    def get_image_height(self):
        return 224

    def get_classes(self) -> list:
        return self.classes

    def __establish_classes(self):
        classes = list(iter(self.TRAINING_BATCHES.class_indices))
        for c in self.TRAINING_BATCHES.class_indices:
            classes[self.TRAINING_BATCHES.class_indices[c]] = c
        self.classes = classes

    def __get_convolutional_layers(self) -> [Sequential]:
        conv_layers = [Lambda(self.__vgg_preprocess, input_shape=(3, self.get_image_width(), self.get_image_height()),
                              output_shape=(3, self.get_image_width(), self.get_image_height()))]
        conv_layers.extend(self.__generate_single_conv_block_layers(2, 64))
        conv_layers.extend(self.__generate_single_conv_block_layers(2, 128))
        conv_layers.extend(self.__generate_single_conv_block_layers(3, 256))
        conv_layers.extend(self.__generate_single_conv_block_layers(3, 512))
        conv_layers.extend(self.__generate_single_conv_block_layers(3, 512))
        return conv_layers


    def __get_dense_layers(self, num_classes: int, drop_out: float) -> [Sequential]:
        return [
            Flatten(),
            Dense(4096, activation='relu'),
            BatchNormalization(),
            Dropout(drop_out),
            Dense(4096, activation='relu'),
            BatchNormalization(),
            Dropout(drop_out),
            Dense(num_classes, activation='softmax')
        ]

    def __generate_original_model(self) -> Sequential:
        model = Sequential()
        conv_layers = self.__get_convolutional_layers()

        for conv_layer in conv_layers:
            model.add(conv_layer)

        dense_layers = self.__get_dense_layers(num_classes=1000, drop_out=self.DROP_OUT)

        for dense_layer in dense_layers:
            model.add(dense_layer)

        model.load_weights(get_file('vgg16_bn.h5', self.ORIGINAL_MODEL_WEIGHTS_URL + 'vgg16_bn.h5', cache_subdir='models'))
        return model

    def __initialize_model(self):
        use_cached_model = self.__can_load_weights_from_cache()

        if use_cached_model:
            self.source_model = self.__load_cached_model()
        else:
            self.source_model = self.__generate_original_model()

        self.source_model.summary()

        source_layers = self.source_model.layers
        last_conv_idx = self.__get_last_conv_index(source_layers)
        source_conv_layers = source_layers[:last_conv_idx + 1]
        source_dense_layers = source_layers[last_conv_idx + 1:]
        self.conv_model_portion = Sequential()

        for layer in source_conv_layers:
            self.conv_model_portion.add(layer)
            layer.trainable = False

        num_classes = self.TRAINING_BATCHES.num_class
        self.dense_model_portion = self.__generate_dense_finetuning_model(num_classes=num_classes, source_conv_layers=source_conv_layers,
                                                                          source_dense_layers=source_dense_layers, using_cached_model=use_cached_model)
        self.model = Sequential()

        for layer in self.conv_model_portion.layers:
            self.model.add(layer)
            layer.trainable = False

        dense_layer_count = 0
        self.dense_model_portion.layers.reverse()

        for layer in self.dense_model_portion.layers:
            if Vgg16.__is_dense_layer(layer):
                dense_layer_count = dense_layer_count + 1

            if dense_layer_count <= self.NUM_DENSE_LAYERS_TO_RETRAIN:
                layer.trainable = True
            else:
                layer.trainable = False

        self.dense_model_portion.layers.reverse()

        for layer in self.dense_model_portion.layers:
            self.model.add(layer)

        self.model.summary()
        Vgg16.__compile(self.model)
        self.__establish_classes()

    @staticmethod
    def __is_dense_layer(layer: Sequential)->bool:
        return type(layer) is Dense

    @staticmethod
    def __is_conv_layer(layer: Sequential)->bool:
        return type(layer) is Convolution2D or type(layer) is Conv2D

    def __load_cached_model(self):
        saved_weights_file_name = self.__get_latest_saved_weights_file_name()

        original_model = self.__generate_original_model()
        original_layers = original_model.layers
        last_conv_idx_original = self.__get_last_conv_index(original_layers)
        original_conv_layers = original_layers[:last_conv_idx_original + 1]
        model = Sequential()

        for layer in original_conv_layers:
            model.add(layer)

        cached_model = load_model(saved_weights_file_name, custom_objects={'__vgg_preprocess': self.__vgg_preprocess})
        cached_model.load_weights(saved_weights_file_name, True)
        cached_layers = cached_model.layers
        last_conv_idx_cached = self.__get_last_conv_index(cached_layers)
        cached_dense_layers = cached_layers[last_conv_idx_cached + 1:]

        for layer in cached_dense_layers:
            model.add(layer)

        return model

    def __get_last_conv_index(self, source_layers):
        conv_layers = [index for index, layer in enumerate(source_layers) if (Vgg16.__is_conv_layer(layer))]
        if len(conv_layers) == 0: return -1
        last_conv_idx = conv_layers[-1]
        return last_conv_idx

    # TODO:  Make dropout configurable
    def __generate_dense_finetuning_model(self, num_classes: int, source_conv_layers: [Sequential],
                    source_dense_layers: [Sequential], using_cached_model: bool) -> Sequential:
        dense_layers = self.__get_dense_layers(num_classes=num_classes, drop_out=self.DROP_OUT)
        model = Sequential()
        model.add(MaxPooling2D(input_shape=source_conv_layers[-1].output_shape[1:]))

        for layer in dense_layers:
            model.add(layer)

        for layer1, layer2 in zip(model.layers, source_dense_layers):
            index = source_dense_layers.index(layer2)
            if index < len(source_dense_layers) - 1 or using_cached_model:
                layer1.set_weights(layer2.get_weights())

        Vgg16.__compile(model)
        return model

    def __get_latest_saved_weights_file_name(self):
        directory = self.CACHE_DIRECTORY
        if not os.path.isdir(directory):
            os.makedirs(directory)

        file_names = glob.glob(directory + "*.h5")
        highest_epoch = 0
        highest_epoch_file = ""

        for file_name in file_names:
            epoch = self.__determine_epoch_num_from_weights_file_name(file_name)
            if epoch > highest_epoch:
                highest_epoch = epoch
                highest_epoch_file = file_name

        return highest_epoch_file

    @staticmethod
    def __determine_epoch_num_from_weights_file_name(file_name: str):
        match_obj = re.match(r'(.*?weights\.)(\d+)(-)(.*?)(-)(.*?)(\.h5)', file_name, re.M | re.I)
        if match_obj:
            return int(match_obj.group(2)) + 1
        return 0

    @staticmethod
    def __generate_single_conv_block_layers(num_layers: int, filters):
        conv_layers = []

        for i in range(num_layers):
            conv_layers.append(ZeroPadding2D((1, 1)))
            conv_layers.append(Conv2D(filters, kernel_size=(3, 3), activation='relu'))

        conv_layers.append(MaxPooling2D((2, 2), strides=(2, 2)))
        return conv_layers

    def __vgg_preprocess(self, x):
        x = x - self.VGG_MEAN
        return x[:, ::-1]  # reverse axis rgb->bgr

    def __can_load_weights_from_cache(self):
        latest_file_name = self.__get_latest_saved_weights_file_name()
        latest_saved_epoch = self.__determine_epoch_num_from_weights_file_name(latest_file_name)
        return self.LOAD_WEIGHTS_FROM_CACHE and latest_saved_epoch > 0

    def __get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical') -> DirectoryIterator:
        return gen.flow_from_directory(path, target_size=(self.get_image_width(), self.get_image_height()), color_mode='rgb',
                                       class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    @staticmethod
    def __compile(model: Sequential):
        # optimizer = Adam(lr=0.001)
        optimizer = RMSprop(lr=0.00001, rho=0.7)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def __fit(self, batches, val_batches, steps_per_epoch: int, nb_epoch=1, initial_epoch=0):
        # tensorBoard = keras.callbacks.TensorBoard(log_dir='./tblogs', histogram_freq=1, write_graph=True, write_images=True)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
        model_checkpoint = keras.callbacks.ModelCheckpoint('./' + self.CACHE_DIRECTORY + '/weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5', monitor='val_loss', verbose=0,
                                                           save_best_only=False,
                                                           save_weights_only=False, mode='auto', period=1)

        # OPTIMIZATION:  First, train the conv model on features, save those, then train fc layer for much faster feedback
        # Requires static images
        if self.FAST_CONV_CACHE_TRAINING:
            conv_cache_directory = self.CACHE_DIRECTORY + '/convcache/'

            conv_cache_training_batches = ConvCacheIterator(cache_directory=conv_cache_directory, batches=batches,
                    batch_id = 'training', conv_model=self.conv_model_portion, batch_size=self.TRAINING_BATCH_SIZE, shuffle=True)
            conv_cache_validation_batches = ConvCacheIterator(cache_directory=conv_cache_directory, batches=val_batches,
                    batch_id = 'validation', conv_model=self.conv_model_portion, batch_size=self.VALIDATION_BATCH_SIZE, shuffle=False)

            self.dense_model_portion.fit_generator(conv_cache_training_batches, steps_per_epoch=steps_per_epoch, epochs=nb_epoch, initial_epoch=initial_epoch,
                                     validation_data=conv_cache_validation_batches, validation_steps=int(np.ceil(val_batches.samples / self.VALIDATION_BATCH_SIZE)),
                                     callbacks=[early_stopping, model_checkpoint])
        else:
            self.model.fit_generator(batches, steps_per_epoch=int(np.ceil(batches.samples / self.TRAINING_BATCH_SIZE)), epochs=nb_epoch, initial_epoch=initial_epoch,
                                     validation_data=val_batches, validation_steps=int(np.ceil(val_batches.samples / self.TRAINING_BATCH_SIZE)),
                                     callbacks=[early_stopping, model_checkpoint])



    def __test(self, path, batch_size=8):
        # noinspection PyTypeChecker
        test_batches = self.__get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, int(np.ceil(test_batches.samples / batch_size)))
