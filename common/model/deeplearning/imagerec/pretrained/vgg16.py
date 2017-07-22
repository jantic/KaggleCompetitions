from __future__ import division, print_function
from __future__ import absolute_import

import json
import numpy as np
import os.path
import glob
import re
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Sequential
from keras.optimizers import Nadam, SGD
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.models import load_model
import keras

from common.model.deeplearning.imagerec.BatchImagePredictionRequestInfo import BatchImagePredictionRequestInfo
from common.model.deeplearning.imagerec.IImageRecModel import IImageRecModel
from common.model.deeplearning.imagerec.ImagePredictionResult import ImagePredictionResult
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from keras.backend import manual_variable_initialization, tf, set_session



class Vgg16(IImageRecModel):
    """The VGG 16 Imagenet model"""

    def __init__(self, loadWeightsFromCache: bool, trainingImagesPath: str, training_batch_size: int, validationImagesPath: str, validation_batch_size: int):
        self.TRAINING_BATCHES = self.__getBatches(trainingImagesPath, batch_size=training_batch_size)
        self.VALIDATION_BATCHES = self.__getBatches(validationImagesPath, batch_size=validation_batch_size)
        self.VGG_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
        self.ORIGINAL_MODEL_WEIGHTS_URL = 'http://www.platform.ai/models/'
        self.LOAD_WEIGHTS_FROM_CACHE = loadWeightsFromCache
        self.LATEST_SAVED_WEIGHTS_FILENAME = self.__getLatestSavedWeightsFileName()
        self.LATEST_SAVED_EPOCH = self.__determineEpochNumFromWeighFileName(self.LATEST_SAVED_WEIGHTS_FILENAME)
        self.__create()
        self.__get_classes()

    def refineTraining(self, numEpochs: int):
        initialEpoch = max(self.LATEST_SAVED_EPOCH,0) if self.LOAD_WEIGHTS_FROM_CACHE else 0
        self.__fit(self.TRAINING_BATCHES, self.VALIDATION_BATCHES, nb_epoch=numEpochs, initial_epoch=initialEpoch)

    def predict(self, imagePredictionRequests: [ImagePredictionRequest], batch_size: int, details=False) -> [ImagePredictionResult]:
        verbose = 1 if details else 0
        batchRequestInfo = BatchImagePredictionRequestInfo.getInstance(imagePredictionRequests, self.getImageWidth(), self.getImageHeight())
        batchConfidences = self.model.predict(batchRequestInfo.getImageArray(), batch_size=batch_size, verbose=verbose)
        imagePredictionResults = ImagePredictionResult.generateImagePredictionResults(batchConfidences, batchRequestInfo, self.classes)
        return imagePredictionResults

    def getImageWidth(self):
        return 224

    def getImageHeight(self):
        return 224

    def __getLatestSavedWeightsFileName(self):
        directory = "./cache/"
        if not os.path.isdir(directory):
            os.mkdir(directory)

        fileNames = glob.glob(directory + "*.h5")
        highestEpoch = -1
        highestEpochFile = ""

        for fileName in fileNames:
            epoch = self.__determineEpochNumFromWeighFileName(fileName)
            if epoch > highestEpoch:
                highestEpoch = epoch
                highestEpochFile = fileName

        return highestEpochFile

    @staticmethod
    def __determineEpochNumFromWeighFileName(fileName):
        matchObj = re.match(r'(.*?weights\.)(\d+)(-)(.*?)(\.h5)', fileName, re.M | re.I)
        if matchObj:
            return int(matchObj.group(2))
        return -1

    def __get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.ORIGINAL_MODEL_WEIGHTS_URL + fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def __ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def __FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

    def __vgg_preprocess(self, x):
        x = x - self.VGG_MEAN
        return x[:, ::-1]  # reverse axis rgb->bgr

    def __create(self):
        if self.LOAD_WEIGHTS_FROM_CACHE and self.LATEST_SAVED_EPOCH > -1:
            self.model = load_model(self.LATEST_SAVED_WEIGHTS_FILENAME, custom_objects={'__vgg_preprocess': self.__vgg_preprocess})
            self.model.load_weights(self.LATEST_SAVED_WEIGHTS_FILENAME, True)
        else:
            self.model = Sequential()
            self.model.add(Lambda(self.__vgg_preprocess, input_shape=(3, self.getImageWidth(), self.getImageHeight()),
                                  output_shape=(3, self.getImageWidth(), self.getImageHeight())))
            self.__ConvBlock(2, 64)
            self.__ConvBlock(2, 128)
            self.__ConvBlock(3, 256)
            self.__ConvBlock(3, 512)
            self.__ConvBlock(3, 512)

            self.model.add(Flatten())
            self.__FCBlock()
            self.__FCBlock()
            self.model.add(Dense(1000, activation='softmax'))
            self.model.load_weights(get_file('vgg16.h5', self.ORIGINAL_MODEL_WEIGHTS_URL, cache_subdir='models'))
            self.__finetune(self.TRAINING_BATCHES)

    def __getBatches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(path, target_size=(self.getImageWidth(), self.getImageHeight()),
                                       class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def __ft(self, num):
        model = self.model
        model.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(num, activation='softmax'))
        self.__compile()

    def __finetune(self, batches):
        self.__ft(batches.nb_class)
        classes = list(iter(batches.class_indices))
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes

    def __compile(self):
        optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def __fitData(self, trn, labels, val, val_labels, nb_epoch=1, batch_size=64):
        self.model.fit(trn, labels, nb_epoch=nb_epoch,
                       validation_data=(val, val_labels), batch_size=batch_size)

    def __fit(self, batches, val_batches, nb_epoch=1, initial_epoch=0):
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')
        modelCheckpoint = keras.callbacks.ModelCheckpoint('./cache/weights.{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', verbose=0, save_best_only=False,
                                                          save_weights_only=False, mode='auto', period=1)
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch, initial_epoch=initial_epoch,
                                 callbacks=[earlyStopping, modelCheckpoint], validation_data=val_batches, nb_val_samples=val_batches.nb_sample)

    def __test(self, path, batch_size=8):
        test_batches = self.__getBatches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)
