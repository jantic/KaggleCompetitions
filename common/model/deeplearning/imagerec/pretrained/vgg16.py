from __future__ import absolute_import
from __future__ import division, print_function

import glob
import os.path
import re

import keras
import numpy as np
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator
from keras.utils.data_utils import get_file

from common.model.deeplearning.imagerec.BatchImagePredictionRequestInfo import BatchImagePredictionRequestInfo
from common.model.deeplearning.imagerec.IImageRecModel import IImageRecModel
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from common.model.deeplearning.imagerec.ImagePredictionResult import ImagePredictionResult


class Vgg16(IImageRecModel):
    """The VGG 16 Imagenet model"""

    def __init__(self, load_weights_from_cache: bool, training_images_path: str, training_batch_size: int, validation_images_path: str, validation_batch_size: int):
        self.TRAINING_BATCH_SIZE = training_batch_size
        self.TRAINING_BATCHES = self.__get_batches(training_images_path, batch_size=training_batch_size)
        self.VALIDATION_BATCHES = self.__get_batches(validation_images_path, batch_size=validation_batch_size)
        self.VGG_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
        self.ORIGINAL_MODEL_WEIGHTS_URL = 'http://files.fast.ai/models/'
        self.LOAD_WEIGHTS_FROM_CACHE = load_weights_from_cache
        self.LATEST_SAVED_WEIGHTS_FILENAME = self.__get_latest_saved_weights_file_name()
        self.LATEST_SAVED_EPOCH = self.__determine_epoch_num_from_weights_file_name(self.LATEST_SAVED_WEIGHTS_FILENAME)
        self.__create()
        self.__establish_classes()

    def refine_training(self, num_epochs: int):
        initial_epoch = max(self.LATEST_SAVED_EPOCH, 0) if self.LOAD_WEIGHTS_FROM_CACHE else 0
        self.__fit(self.TRAINING_BATCHES, self.VALIDATION_BATCHES, nb_epoch=num_epochs, initial_epoch=initial_epoch)

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

    def __generateFreshKerasModel(self) -> Sequential:
        model = Sequential()
        model.add(Lambda(self.__vgg_preprocess, input_shape=(3, self.get_image_width(), self.get_image_height()),
                         output_shape=(3, self.get_image_width(), self.get_image_height())))
        Vgg16.__conv_block(model, 2, 64)
        Vgg16.__conv_block(model, 2, 128)
        Vgg16.__conv_block(model, 3, 256)
        Vgg16.__conv_block(model, 3, 512)
        Vgg16.__conv_block(model, 3, 512)

        model.add(Flatten())
        Vgg16.__fc_block(model)
        Vgg16.__fc_block(model)
        model.add(Dense(1000, activation='softmax'))
        model.load_weights(get_file('vgg16.h5', self.ORIGINAL_MODEL_WEIGHTS_URL + 'vgg16.h5', cache_subdir='models'))
        self.__finetune(model)
        return model

    def __get_latest_saved_weights_file_name(self):
        directory = "./cache/"
        if not os.path.isdir(directory):
            os.mkdir(directory)

        file_names = glob.glob(directory + "*.h5")
        highest_epoch = 0
        highest_epoch_file = ""

        for fileName in file_names:
            epoch = self.__determine_epoch_num_from_weights_file_name(fileName)
            if epoch > highest_epoch:
                highest_epoch = epoch
                highest_epoch_file = fileName

        return highest_epoch_file

    @staticmethod
    def __determine_epoch_num_from_weights_file_name(fileName):
        match_obj = re.match(r'(.*?weights\.)(\d+)(-)(.*?)(-)(.*?)(\.h5)', fileName, re.M | re.I)
        if match_obj:
            return int(match_obj.group(2)) + 1
        return 0

    @staticmethod
    def __conv_block(model: Sequential, layers, filters):
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    @staticmethod
    def __fc_block(model: Sequential):
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

    def __vgg_preprocess(self, x):
        x = x - self.VGG_MEAN
        return x[:, ::-1]  # reverse axis rgb->bgr

    def __can_load_weights_from_cache(self):
        return self.LOAD_WEIGHTS_FROM_CACHE and self.LATEST_SAVED_EPOCH > 0

    def __create(self):
        if self.__can_load_weights_from_cache():
            self.model = load_model(self.LATEST_SAVED_WEIGHTS_FILENAME, custom_objects={'__vgg_preprocess': self.__vgg_preprocess})
            self.model.load_weights(self.LATEST_SAVED_WEIGHTS_FILENAME, True)
        else:
            self.model = self.__generateFreshKerasModel()

    def __get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical') -> DirectoryIterator:
        return gen.flow_from_directory(path, target_size=(self.get_image_width(), self.get_image_height()), color_mode='rgb',
                                       class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def __finetune(self, model: Sequential):
        num_classes = self.TRAINING_BATCHES.num_class
        model.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(num_classes, activation='softmax'))
        Vgg16.__compile(model)

    @staticmethod
    def __compile(model: Sequential):
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def __fit(self, batches, val_batches, nb_epoch=1, initial_epoch=0):
        # tensorBoard = keras.callbacks.TensorBoard(log_dir='./tblogs', histogram_freq=1, write_graph=True, write_images=True)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
        model_checkpoint = keras.callbacks.ModelCheckpoint('./cache/weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5', monitor='val_loss', verbose=0, save_best_only=False,
                                                           save_weights_only=False, mode='auto', period=1)
        self.model.fit_generator(batches, steps_per_epoch=int(np.ceil(batches.samples / self.TRAINING_BATCH_SIZE)), epochs=nb_epoch, initial_epoch=initial_epoch,
                                 validation_data=val_batches, validation_steps=int(np.ceil(val_batches.samples / self.TRAINING_BATCH_SIZE)),
                                 callbacks=[early_stopping, model_checkpoint])

    def __test(self, path, batch_size=8):
        # noinspection PyTypeChecker
        test_batches = self.__get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, int(np.ceil(test_batches.samples / batch_size)))
