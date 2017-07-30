from __future__ import division, print_function
import numpy as np
from importlib import reload
from common.utils import utils
from common.model.deeplearning.imagerec.pretrained import vgg16
from common.setup.DataSetup import DataSetup
from common.model.deeplearning.imagerec.pretrained.vgg16 import Vgg16
from common.output.csv.KaggleCsvWriter import KaggleCsvWriter
from common.visualization.ImagePerformanceVisualizer import ImagePerformanceVisualizer
from common.model.deeplearning.imagerec.MasterImageClassifier import MasterImageClassifier

runMainTest = False
refineTraining = False
imageSplitting = False
initializeData = False
visualizePerformance = True
visualizationClass = 'c5'
useSample = False
numberOfEpochs = 30
training_batch_size = 64
validation_batch_size = 64
test_batch_size = 64

reload(utils)
np.set_printoptions(precision=4, linewidth=100)
reload(vgg16)

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

vgg = Vgg16(True, trainingSetPath, training_batch_size, validationSetPath, validation_batch_size)

if refineTraining:
    vgg.refineTraining(numberOfEpochs)

imageClassifier = MasterImageClassifier(vgg)

if runMainTest:
    predictionSummaries = imageClassifier.getAllPredictions(mainTestSetPath, False, test_batch_size)
    KaggleCsvWriter.writePredictionsForClassIdToCsv(predictionSummaries, 1)

if visualizePerformance:
    testResultSummaries = []
    classes = vgg.getClasses()

    for clazz in classes:
        visualizationTestPath = validationSetPath + "/" + clazz
        testResultSummaries.extend(imageClassifier.getAllTestResults(visualizationTestPath, imageSplitting, test_batch_size, clazz))

    ImagePerformanceVisualizer.doVisualizations(testResultSummaries, visualizationClass, 5, True, True, True, True)
