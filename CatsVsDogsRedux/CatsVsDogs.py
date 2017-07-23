from __future__ import division, print_function
import numpy as np
from importlib import reload
from common.utils import utils
from common.model.deeplearning.imagerec.pretrained import vgg16
from common.model.deeplearning.imagerec.pretrained.vgg16 import Vgg16
from common.output.csv.KaggleCsvWriter import KaggleCsvWriter
from common.model.deeplearning.imagerec.MasterImageClassifier import MasterImageClassifier
import cProfile
import time


reload(utils)
np.set_printoptions(precision=4, linewidth=100)
reload(vgg16)

dataPath = "data/"
#dataPath = "data/sample/"
trainingSetPath = dataPath + "train"
validationSetPath = dataPath + "valid"
testSetPath = dataPath + "test1"
numberOfEpochs = 50
training_batch_size = 64
validation_batch_size = 64
test_batch_size = 64
vgg = Vgg16(True, trainingSetPath, training_batch_size, validationSetPath, validation_batch_size)
vgg.refineTraining(numberOfEpochs)
imageClassifer = MasterImageClassifier(vgg)

#pr = cProfile.Profile()
#pr.enable()
start = time.time()
predictionSummaries = imageClassifer.getAllPredictions(testSetPath, False, test_batch_size)
end = time.time()
print(end - start)
#pr.disable()
#pr.print_stats(sort="cumtime")
KaggleCsvWriter.writePredictionsForClassIdToCsv(predictionSummaries, 1)
