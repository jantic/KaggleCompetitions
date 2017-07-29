from __future__ import division, print_function
import numpy as np
from importlib import reload

from common.image.ImageInfo import ImageInfo
from common.utils import utils
from common.model.deeplearning.imagerec.pretrained import vgg16
import os
from common.setup.DataSetup import DataSetup
from common.model.deeplearning.imagerec.pretrained.vgg16 import Vgg16
from common.output.csv.KaggleCsvWriter import KaggleCsvWriter
from common.visualization.ImagePerformanceVisualizer import ImagePerformanceVisualizer
from common.model.deeplearning.imagerec.MasterImageClassifier import MasterImageClassifier
import cProfile
import time

reload(utils)
np.set_printoptions(precision=4, linewidth=100)
reload(vgg16)


initializeData = False
useSample = False

numberOfEpochs = 30
training_batch_size = 64
validation_batch_size = 64
test_batch_size = 64
visualizationClass = 'c0'

mainDataPath = "data/main/"
mainTrainingSetPath = mainDataPath + "train"
mainValidationSetPath = mainDataPath + "valid"
mainTestSetPath = mainDataPath + "test1"

sampleDataPath = "data/sample/"
sampleTrainingSetPath = sampleDataPath + "train"
sampleValidationSetPath = sampleDataPath + "valid"
sampleTestSetPath = sampleDataPath + "test1"

if initializeData:
    DataSetup.establishValidationDataIfNeeded(mainTrainingSetPath, mainValidationSetPath)
    DataSetup.establishSampleDataIfNeeded(mainDataPath, sampleDataPath, sampleRatio=0.04)

trainingSetPath = sampleTrainingSetPath if useSample else mainTrainingSetPath
validationSetPath = sampleValidationSetPath if useSample else mainValidationSetPath
visualizationTestPath = validationSetPath + "/" + visualizationClass

vgg = Vgg16(True, trainingSetPath, training_batch_size, validationSetPath, validation_batch_size)
vgg.refineTraining(numberOfEpochs)
imageClassifier = MasterImageClassifier(vgg)

#######END CORE#################

# predictionSummaries = imageClassifer.getAllPredictions(testSetPath, False, test_batch_size)
# KaggleCsvWriter.writePredictionsForClassIdToCsv(predictionSummaries, 1)
testResultSummaries = imageClassifier.getAllTestResults(visualizationTestPath, False, test_batch_size, visualizationClass)
ImagePerformanceVisualizer.doVisualizations(testResultSummaries, visualizationClass, 5, True, True, True)
