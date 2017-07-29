import shutil
from glob import glob
from numpy.random import permutation
import os


class DataSetup:
    @staticmethod
    def establishValidationDataIfNeeded(trainingDirectory: str, validDirectory: str, imageFileExtension='jpg', validToTestRatio=0.1):
        trainingDirectory = str.replace(trainingDirectory, "/", "\\")
        validDirectory = str.replace(validDirectory, "/", "\\")

        if not os.path.exists(validDirectory):
            os.mkdir(validDirectory)

        validationClassDirectoryNames = glob(validDirectory + "/*/")

        if len(validationClassDirectoryNames) > 0:
            return

        trainingClassDirectoryPaths = glob(trainingDirectory + "/*/")

        for trainingClassDirectoryPath in trainingClassDirectoryPaths:
            newValidClassDirectoryPath = str.replace(trainingClassDirectoryPath, trainingDirectory, validDirectory)

            if not os.path.exists(newValidClassDirectoryPath):
                os.mkdir(newValidClassDirectoryPath)

            trainingImages = glob(trainingClassDirectoryPath + "/*." + imageFileExtension)
            numValidImages = int(round(validToTestRatio * len(trainingImages), 0))
            testImagesToMove = permutation(trainingImages)[:numValidImages]

            for testImageToMove in testImagesToMove:
                newValidImagePath = str.replace(testImageToMove, trainingDirectory, validDirectory)
                shutil.move(testImageToMove, newValidImagePath)

    @staticmethod
    def establishSampleDataIfNeeded(mainDataDirectory: str, sampleDirectory: str, imageFileExtension='jpg', sampleRatio=0.02):
        mainDataDirectory = str.replace(mainDataDirectory, "/", "\\")
        sampleDirectory = str.replace(sampleDirectory, "/", "\\")

        if not os.path.exists(sampleDirectory):
            os.mkdir(sampleDirectory)

        sampleDirectoryNames = glob(sampleDirectory + "/*/")

        if len(sampleDirectoryNames) > 0:
            return

        mainDataDirectoryPaths = glob(mainDataDirectory + "/*/")

        for mainDataDirectoryPath in mainDataDirectoryPaths:
            newSampleDirectoryPath = str.replace(mainDataDirectoryPath, mainDataDirectory, sampleDirectory)

            if not os.path.exists(newSampleDirectoryPath):
                os.mkdir(newSampleDirectoryPath)

            mainSubDirectoryPaths = glob(mainDataDirectoryPath + "/*/")

            for mainSubDirectoryPath in mainSubDirectoryPaths:
                newSampleSubDirectoryPath = str.replace(mainSubDirectoryPath, mainDataDirectory, sampleDirectory)

                if not os.path.exists(newSampleSubDirectoryPath):
                    os.mkdir(newSampleSubDirectoryPath)

                sourceImages = glob(mainSubDirectoryPath + "/*." + imageFileExtension)
                numSampleImages = int(round(sampleRatio * len(sourceImages), 0))
                sourceImagesToCopy = permutation(sourceImages)[:numSampleImages]

                for sourceImageToCopy in sourceImagesToCopy:
                    newSampleImagePath = str.replace(sourceImageToCopy, mainDataDirectory, sampleDirectory)
                    shutil.copy(sourceImageToCopy, newSampleImagePath)


