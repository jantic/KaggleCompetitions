from __future__ import division, print_function
from __future__ import absolute_import

import json

import numpy as np
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils.data_utils import get_file
import keras

from common.model.deeplearning.imagerec.IImageRecModel import IImageRecModel
from common.model.deeplearning.imagerec.ImagePredictionResult import ImagePredictionResult
from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary
from common.model.deeplearning.prediction.PredictionInfo import PredictionInfo
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest
from common.image.ModelImageConverter import ModelImageConverter
from common.image.ImageInfo import ImageInfo

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr


class Vgg16(IImageRecModel):
    """The VGG 16 Imagenet model"""


    def __init__(self):
        self.FILE_PATH = 'http://www.platform.ai/models/'
        self.create()
        self.get_classes()


    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def getImageWidth(self):
        return 224

    def getImageHeight(self):
        return 224

    def getMinConfidence(self):
        return 0.02

    def getMaxConfidence(self):
        return 0.98

    #TODO:  Clean this up and break it up into smaller more understandable functions!
    def predict(self, imagePredictionRequests : [ImagePredictionRequest], batch_size : int, details=False) -> [ImagePredictionResult]:
        verbose = 1 if details else 0

        testIdToOrderedImageInfos = self.__generateTestIdToOrderedImageInfosMapping(imagePredictionRequests)
        testIdToPredictionSummaries = {}

        while(True):
            batchTestIds, batchImageInfos = self.__generateNextBatchData(testIdToOrderedImageInfos, batch_size)
            if len(batchTestIds) == 0 : break
            batchPilImages = ModelImageConverter.getAllPilImages(batchImageInfos)
            imageArray = ModelImageConverter.generateImageArrayForPrediction(batchPilImages, self.getImageWidth(), self.getImageHeight())
            confidenceBatches = self.model.predict(imageArray, verbose=verbose)
            self.__populateWithBatchResults(testIdToPredictionSummaries, confidenceBatches, batchTestIds, batchImageInfos)

        imagePredictionResults = self.__generateImagePredictionResults(testIdToPredictionSummaries)
        return imagePredictionResults

    def __generateImagePredictionResults(self, testIdToPredictionSummaries : {}) -> [ImagePredictionResult]:
        imagePredictionResults = []

        for testId in testIdToPredictionSummaries.keys():
            predictionSummaries = testIdToPredictionSummaries[testId]
            imagePredictionResult = ImagePredictionResult.getInstance(testId, predictionSummaries)
            imagePredictionResults.append(imagePredictionResult)

        return imagePredictionResults

    def __populateWithBatchResults(self, testIdToPredictionSummaries : {}, confidenceBatches : [float], batchTestIds : [int], batchImageInfos : [ImageInfo]):
        for index in range(len(confidenceBatches)):
            testId = batchTestIds[index]
            imageInfo = batchImageInfos[index]
            confidences = confidenceBatches[index]
            predictionSummary = self.__generatePredictionSummary(testId, imageInfo, confidences)

            if not (testId in testIdToPredictionSummaries):
                testIdToPredictionSummaries[testId] = []

            testIdToPredictionSummaries[testId].append(predictionSummary)

    def __generateNextBatchData(self, testIdToOrderedImageInfos : {}, batch_size : int):
        batchTestIds = []
        batchImageInfos = []
        previousTestId = None

        while(len(batchTestIds) < batch_size and (len(testIdToOrderedImageInfos.keys()) > 0)):
            currentTestId = self.__getCurrentTestId(testIdToOrderedImageInfos, previousTestId)
            if currentTestId == None: break
            self.__prepareDataForForBatch(testIdToOrderedImageInfos, currentTestId, batchTestIds, batchImageInfos, batch_size)
            previousTestId = currentTestId

        return batchTestIds, batchImageInfos

    def __getCurrentTestId(self, testIdToOrderedImageInfos : {}, previousTestId : int) -> int:
        if previousTestId == None:
            return list(testIdToOrderedImageInfos.keys())[0]

        if not previousTestId in testIdToOrderedImageInfos:
            if len(testIdToOrderedImageInfos.keys()) > 0:
                return list(testIdToOrderedImageInfos.keys())[0]
            else:
                return None
        else:
            return previousTestId

    def __prepareDataForForBatch(self, testIdToOrderedImageInfos : {}, currentTestId : int, batchTestIds : [int], batchImageInfos : [ImageInfo], batch_size : int):
        while len(testIdToOrderedImageInfos[currentTestId]) > 0 and len(batchTestIds) < batch_size:
            imageInfo = testIdToOrderedImageInfos[currentTestId].pop(0)
            batchImageInfos.append(imageInfo)
            batchTestIds.append(currentTestId)

        if len(testIdToOrderedImageInfos[currentTestId]) == 0:
            del testIdToOrderedImageInfos[currentTestId]

    def __generatePredictionSummary(self, testId : int, imageInfo : ImageInfo, confidences : [float]) -> PredictionsSummary:
        classIds = range(len(confidences))
        classNames = [self.classes[classId] for classId in classIds]
        predictionInfos = PredictionInfo.generatePredictionInfos(
            confidences, classIds, classNames, self.getMinConfidence(), self.getMaxConfidence())
        predictionSummary = PredictionsSummary(imageInfo, predictionInfos)
        return predictionSummary

    def __generateTestIdToOrderedImageInfosMapping(self, imagePredictionRequests : [ImagePredictionRequest]) -> {}:
        testIdToOrderedImageInfos = {}

        for imagePredictionRequest in imagePredictionRequests:
            imageInfos = imagePredictionRequest.getImageInfos()
            testId = imagePredictionRequest.getTestId()
            if not(testId in testIdToOrderedImageInfos):
                testIdToOrderedImageInfos[testId] = []

            testIdToOrderedImageInfos[testId].extend(imageInfos)

        return testIdToOrderedImageInfos

    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))


    def create(self):
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,self.getImageWidth(),self.getImageHeight()),
                         output_shape=(3,self.getImageWidth(),self.getImageHeight())))
        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))


    def getBatches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(path, target_size=(self.getImageWidth(),self.getImageHeight()),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


    def ft(self, num):
        model = self.model
        model.pop()
        for layer in model.layers: layer.trainable=False
        model.add(Dense(num, activation='softmax'))
        self.compile()

    def finetune(self, batches):
        self.ft(batches.nb_class)
        classes = list(iter(batches.class_indices))
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes


    def compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])


    def fitData(self, trn, labels,  val, val_labels,  nb_epoch=1, batch_size=64):
        self.model.fit(trn, labels, nb_epoch=nb_epoch,
                validation_data=(val, val_labels), batch_size=batch_size)


    def fit(self, batches, val_batches, nb_epoch=1):
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1, mode='auto')
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                callbacks = [earlyStopping], validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


    def test(self, path, batch_size=8):
        test_batches = self.getBatches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)

