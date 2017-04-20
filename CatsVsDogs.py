from __future__ import division, print_function

import os, json
import glob
import numpy as np

np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

from importlib import reload
import utils;

reload(utils)
from utils import plots
import vgg16;
reload(vgg16)
from vgg16 import Vgg16
import matplotlib
import pandas as pd

from ImageInfo import ImageInfo
from PredictionInfo import PredictionInfo
from PredictionsSummary import PredictionsSummary
from KaggleCsvWriter import KaggleCsvWriter

def getAllPredictions(imageInfos, vgg):
    predictionSummaries = []

    for imageInfo in imageInfos:
        images = [imageInfo.getImageArray()]
        probs, classIds, classNames = vgg.predict(images)
        classId = classIds[0]
        className = classNames[0]
        confidence = probs[0]
        testId = imageInfo.getImageNumber()
        topPredictionInfo = PredictionInfo(confidence, classId, className)
        secondPredictionInfo = generateSecondPredictionInfo(topPredictionInfo, vgg)
        predictionInfos = [topPredictionInfo, secondPredictionInfo]
        predictionSummary = PredictionsSummary(testId, predictionInfos)
        predictionSummaries.append(predictionSummary)

    return predictionSummaries

def generateSecondPredictionInfo(topPredictionInfo, vgg):
    classId = 1 if (topPredictionInfo.getClassId() == 0) else 0
    className = vgg.classes[classId]
    confidence = 1- topPredictionInfo.getConfidence()
    secondPredictionInfo = PredictionInfo(confidence, classId, className)
    return secondPredictionInfo

dataPath = "data/"
trainingSetPath = dataPath + "train"
validationSetPath = dataPath + "valid"
testSetPath = dataPath + "test1"
nb_epoch = 2
training_batch_size = 64
validation_batch_size = 80
#TODO Toggle size?
image_size = [224, 224]
vgg = Vgg16()

trainingBatches = vgg.get_batches(trainingSetPath, batch_size=training_batch_size)
validationBatches = vgg.get_batches(validationSetPath, batch_size=validation_batch_size)
vgg.finetune(trainingBatches)
vgg.fit(trainingBatches, validationBatches, nb_epoch=nb_epoch)

testImageInfos = ImageInfo.loadImageInfosFromDirectory(testSetPath, image_size)

predictionSummaries = getAllPredictions(testImageInfos, vgg)
csvWriter = KaggleCsvWriter()
csvWriter.writePredictionsForClassIdToCsv(predictionSummaries, 1)







