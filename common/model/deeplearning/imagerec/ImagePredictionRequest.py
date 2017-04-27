from common.image.ImageInfo import ImageInfo

class ImagePredictionRequest:
    @staticmethod
    def generateInstances(imageInfos : [ImageInfo]) -> []:
        groupedImageInfos = {}

        for imageInfo in imageInfos:
            testId = imageInfo.getImageNumber()
            if not(testId in groupedImageInfos):
                groupedImageInfos[testId] = []
            groupedImageInfos[testId].append(imageInfo)

        requests = []

        for imageInfoGroup in groupedImageInfos.values():
            request = ImagePredictionRequest(imageInfoGroup)
            requests.append(request)

        return requests


    def __init__(self, imageInfos : [ImageInfo]):
        self.__imageInfos = imageInfos
        self.__testId = imageInfos[0].getImageNumber()

    def getImageInfos(self):
        return self.__imageInfos

    def getTestId(self):
        return self.__testId