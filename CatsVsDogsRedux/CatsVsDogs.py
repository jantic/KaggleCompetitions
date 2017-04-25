from __future__ import division, print_function

import numpy as np

np.set_printoptions(precision=4, linewidth=100)

from importlib import reload
from common.utils import utils
reload(utils)

from common.model.deeplearning.imagerec.pretrained import vgg16

reload(vgg16)
from common.model.deeplearning.imagerec.pretrained.vgg16 import Vgg16
from common.output.csv.KaggleCsvWriter import KaggleCsvWriter
from common.model.deeplearning.imagerec.MasterImageClassifier import MasterImageClassifier

dataPath = "data/sample/"
trainingSetPath = dataPath + "train"
validationSetPath = dataPath + "valid"
testSetPath = dataPath + "test1"
numberOfEpochs = 1
training_batch_size = 64
validation_batch_size = 80
vgg = Vgg16()

imageClassifer = MasterImageClassifier(vgg)
imageClassifer.refineTraining(trainingSetPath, training_batch_size, validationSetPath, validation_batch_size, numberOfEpochs)
predictionSummaries = imageClassifer.getAllPredictions(testSetPath)
csvWriter = KaggleCsvWriter()
csvWriter.writePredictionsForClassIdToCsv(predictionSummaries, 1)
