from __future__ import division, print_function
import numpy as np
from importlib import reload
from common.utils import utils
from common.model.deeplearning.imagerec.pretrained import vgg16
from common.model.deeplearning.imagerec.pretrained.vgg16 import Vgg16
from common.output.csv.KaggleCsvWriter import KaggleCsvWriter
from common.model.deeplearning.imagerec.MasterImageClassifier import MasterImageClassifier
reload(utils)
np.set_printoptions(precision=4, linewidth=100)
reload(vgg16)


dataPath = "data/"
#dataPath = "data/sample/"
trainingSetPath = dataPath + "train"
validationSetPath = dataPath + "valid"
testSetPath = dataPath + "test1"
numberOfEpochs = 10
training_batch_size = 64
validation_batch_size = 80
test_batch_size = 80
vgg = Vgg16(True)
vgg.refineTraining(trainingSetPath, training_batch_size, validationSetPath, validation_batch_size, numberOfEpochs)
imageClassifer = MasterImageClassifier(vgg)
predictionSummaries = imageClassifer.getAllPredictions(testSetPath, False, test_batch_size)
KaggleCsvWriter.writePredictionsForClassIdToCsv(predictionSummaries, 1)
