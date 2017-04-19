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
        probs, idxs, classes = vgg.predict(imageInfo.getImageArray())
        predictionInfos = PredictionInfo.generatePredictionInfos(probs, idxs, classes)
        predictionSummary = PredictionsSummary(imageInfo.getImageNumber(), predictionInfos)
        predictionSummaries.append(predictionSummary)

    return predictionSummaries


dataPath = "data/sample/"
nb_epoch = 1
batch_size = 40
vgg = Vgg16()

trainingBatches = vgg.get_batches(dataPath + 'train', batch_size=batch_size)
validationBatches = vgg.get_batches(dataPath + 'valid', batch_size=batch_size * 2)
vgg.finetune(trainingBatches)
vgg.fit(trainingBatches, validationBatches, nb_epoch=nb_epoch)
imageInfos = ImageInfo.loadImageInfosFromDirectory(dataPath + "test1", [224, 224], "jpg")
predictionSummaries = getAllPredictions(imageInfos, vgg)
csvWriter = KaggleCsvWriter()
csvWriter.writePredictionsToCsv(predictionSummaries)







