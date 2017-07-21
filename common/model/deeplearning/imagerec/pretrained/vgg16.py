from __future__ import division, print_function
from __future__ import absolute_import

import json
import numpy as np
import os.path
import time
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils.data_utils import get_file
import keras

from common.model.deeplearning.imagerec.BatchImagePredictionRequestInfo import BatchImagePredictionRequestInfo
from common.model.deeplearning.imagerec.IImageRecModel import IImageRecModel
from common.model.deeplearning.imagerec.ImagePredictionResult import ImagePredictionResult
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))


def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1]  # reverse axis rgb->bgr


class Vgg16(IImageRecModel):
    """The VGG 16 Imagenet model"""

    def __init__(self, loadWeightsFromCache: bool):
        self.ORIGINAL_MODEL_WEIGHTS_URL = 'http://www.platform.ai/models/'
        self.__create(loadWeightsFromCache)
        self.__get_classes()

    def refineTraining(self, trainingImagesPath: str, training_batch_size: int, validationImagesPath: str, validation_batch_size: int, numEpochs: int):
        trainingBatches = self.__getBatches(trainingImagesPath, batch_size=training_batch_size)
        validationBatches = self.__getBatches(validationImagesPath, batch_size=validation_batch_size)
        self.__finetune(trainingBatches)
        startingEpoch = self.__getLatestCachedEpochNum() + 1

        for epoch in range(startingEpoch, numEpochs):
            if epoch > 2:
                self.model.optimizer.lr = 0.01

            print('-------------Actual Epoch: ' + str(epoch) + ' of ' + str(numEpochs) + ' epochs--------------')
            self.__fit(trainingBatches, validationBatches, nb_epoch=1)
            self.__saveWeights(epoch)

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

    def __generateEpochWeightsCachePath(self, epochNum: int):
        dir = "./cache"
        if not os.path.isdir(dir):
            os.mkdir(dir)

        return dir + "/vgg16_epoch_" + str(epochNum) + ".h5"

    def __getLatestCachedEpochNum(self):
        epochNum = 1

        while True:
            if not os.path.isfile(self.__generateEpochWeightsCachePath(epochNum)):
                return epochNum -1
            epochNum = epochNum + 1


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

    def __create(self, loadWeightsFromCache: bool):
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3, self.getImageWidth(), self.getImageHeight()),
                         output_shape=(3, self.getImageWidth(), self.getImageHeight())))
        self.__ConvBlock(2, 64)
        self.__ConvBlock(2, 128)
        self.__ConvBlock(3, 256)
        self.__ConvBlock(3, 512)
        self.__ConvBlock(3, 512)

        model.add(Flatten())
        self.__FCBlock()
        self.__FCBlock()
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16.h5'

        latestCachedEpoch = self.__getLatestCachedEpochNum()

        if loadWeightsFromCache and latestCachedEpoch > 0:
            cachedPath = self.__generateEpochWeightsCachePath(latestCachedEpoch)
            model.load_weights(cachedPath, True)
        else:
            model.load_weights(get_file(fname, self.ORIGINAL_MODEL_WEIGHTS_URL, cache_subdir='models'))

    def __getBatches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(path, target_size=(self.getImageWidth(), self.getImageHeight()),
                                       class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def __ft(self, num):
        model = self.model
        model.pop()
        for layer in model.layers: layer.trainable = False
        model.add(Dense(num, activation='softmax'))
        self.__compile()

    def __finetune(self, batches):
        self.__ft(batches.nb_class)
        classes = list(iter(batches.class_indices))
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes

    def __compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    def __fitData(self, trn, labels, val, val_labels, nb_epoch=1, batch_size=64):
        self.model.fit(trn, labels, nb_epoch=nb_epoch,
                       validation_data=(val, val_labels), batch_size=batch_size)

    def __fit(self, batches, val_batches, nb_epoch=1):
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1, mode='auto')
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                                 callbacks=[earlyStopping], validation_data=val_batches, nb_val_samples=val_batches.nb_sample)

    def __test(self, path, batch_size=8):
        test_batches = self.__getBatches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)

    def __saveWeights(self, epochNum: int):
        filePath = self.__generateEpochWeightsCachePath(epochNum)
        try:
            self.model.save(filePath)
        except AttributeError as error:
            print(error)


