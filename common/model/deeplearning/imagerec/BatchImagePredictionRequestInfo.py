from common.image.ImageInfo import ImageInfo
from common.image.ModelImageConverter import ModelImageConverter
from common.model.deeplearning.imagerec.ImagePredictionRequest import ImagePredictionRequest


class BatchImagePredictionRequestInfo:

    @staticmethod
    def getInstance(imagePredictionRequests : [ImagePredictionRequest], targetImageWidth : int, targetImageHeight : int):
        testIdToOrderedImageInfos = BatchImagePredictionRequestInfo.__generateTestIdToOrderedImageInfosMapping(imagePredictionRequests)
        testIds, imageInfos = BatchImagePredictionRequestInfo.__generateBatchData(testIdToOrderedImageInfos)
        batchPilImages = ModelImageConverter.getAllPilImages(imageInfos)
        imageArray = ModelImageConverter.generateImageArrayForPrediction(batchPilImages, targetImageWidth, targetImageHeight)
        return BatchImagePredictionRequestInfo(testIds, imageInfos, imageArray)

    @staticmethod
    def __generateTestIdToOrderedImageInfosMapping(imagePredictionRequests : [ImagePredictionRequest]) -> {}:
        testIdToOrderedImageInfos = {}

        for imagePredictionRequest in imagePredictionRequests:
            imageInfos = imagePredictionRequest.getImageInfos()
            testId = imagePredictionRequest.getTestId()
            if not(testId in testIdToOrderedImageInfos):
                testIdToOrderedImageInfos[testId] = []

            testIdToOrderedImageInfos[testId].extend(imageInfos)

        return testIdToOrderedImageInfos

    @staticmethod
    def __generateBatchData(testIdToOrderedImageInfos : {}):
        batchTestIds = []
        batchImageInfos = []


        for testId in testIdToOrderedImageInfos.keys():
            BatchImagePredictionRequestInfo.__prepareDataForForBatch(testIdToOrderedImageInfos, testId, batchTestIds, batchImageInfos)

        return batchTestIds, batchImageInfos

    @staticmethod
    def __prepareDataForForBatch(testIdToOrderedImageInfos : {}, currentTestId : int, batchTestIds : [int], batchImageInfos : [ImageInfo]):
        for imageInfo in testIdToOrderedImageInfos[currentTestId]:
            batchImageInfos.append(imageInfo)
            batchTestIds.append(currentTestId)

    def __init__(self, testIds : [int], imageInfos : [ImageInfo], imageArray : [int]):
        self.__testIds = testIds
        self.__imageInfos = imageInfos
        self.__imageArray = imageArray

    def getImageArray(self):
        return self.__imageArray

    def getTestIds(self):
        return self.__testIds

    def getImageInfos(self):
        return self.__imageInfos