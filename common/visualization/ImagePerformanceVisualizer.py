from common.model.deeplearning.test.TestResultSummary import TestResultSummary
import numpy as np
from matplotlib import pyplot as plt


class ImagePerformanceVisualizer:
    @staticmethod
    def doVisualizations(testResultSummaries: [TestResultSummary], className: str, numTestsToVisualize: int, visualizeSuccess: bool,
                         visualizeFailure: bool, visualizeLeastConfident: bool):
        if visualizeSuccess:
            ImagePerformanceVisualizer.__visualizeMostConfidentSuccessfulPredictions(testResultSummaries, className, numTestsToVisualize)

        if visualizeFailure:
            ImagePerformanceVisualizer.__visualizeMostConfidentFailingPredictions(testResultSummaries, className, numTestsToVisualize)

        if visualizeLeastConfident:
            ImagePerformanceVisualizer.__visualizeLeastConfidentPredictions(testResultSummaries, className, numTestsToVisualize)

        plt.show()

    @staticmethod
    def __visualizeMostConfidentSuccessfulPredictions(testResultSummaries: [TestResultSummary], className: str, numTestsToVisualize: int):
        if len(testResultSummaries) == 0 or numTestsToVisualize < 1:
            return

        correct = [testResultSummary for testResultSummary in testResultSummaries if (testResultSummary.isCorrect() and
                                                                                      testResultSummary.getActualClassName() == className)]

        sortedCorrect = sorted(correct, key=ImagePerformanceVisualizer.__confidenceSortKey, reverse=True)

        toDisplay = sortedCorrect[:numTestsToVisualize]
        ImagePerformanceVisualizer.__prepareVisualizations(toDisplay, 'Most Confident Successful Predictions')

    @staticmethod
    def __visualizeMostConfidentFailingPredictions(testResultSummaries: [TestResultSummary], className: str, numTestsToVisualize: int):
        if len(testResultSummaries) == 0 or numTestsToVisualize < 1:
            return

        failing = [testResultSummary for testResultSummary in testResultSummaries if (not testResultSummary.isCorrect() and
                                                                                      testResultSummary.getActualClassName() == className)]

        sortedFailing = sorted(failing, key=ImagePerformanceVisualizer.__confidenceSortKey, reverse=True)

        toDisplay = sortedFailing[:numTestsToVisualize]
        ImagePerformanceVisualizer.__prepareVisualizations(toDisplay, 'Most Confident Failing Predictions')

    @staticmethod
    def __visualizeLeastConfidentPredictions(testResultSummaries: [TestResultSummary], className: str, numTestsToVisualize: int):
        if len(testResultSummaries) == 0 or numTestsToVisualize < 1:
            return

        testResultsOfClass = [testResultSummary for testResultSummary in testResultSummaries if testResultSummary.getActualClassName() == className]

        sortedResults = sorted(testResultsOfClass, key=ImagePerformanceVisualizer.__confidenceSortKey, reverse=False)
        toDisplay = sortedResults[:numTestsToVisualize]
        ImagePerformanceVisualizer.__prepareVisualizations(toDisplay, 'Least Confident Predictions')

    @staticmethod
    def __prepareVisualizations(testResultSummaries: [TestResultSummary], summaryTitle: str):
        images = [testResultSummary.getPredictionSummary().getImageInfo().getPilImage() for testResultSummary in testResultSummaries]
        titles = [testResultSummary.getPredictionSummary().getTopPrediction().getClassName()
                  + ': ' + str(testResultSummary.getPredictionSummary().getTopPrediction().getConfidence()) for testResultSummary in testResultSummaries]
        figsize = (12, 6)
        rows = 1
        interp = False

        plt.interactive(False)
        if type(images[0]) is np.ndarray:
            images = np.array(images).astype(np.uint8)
            if images.shape[-1] != 3:
                images = images.transpose((0, 2, 3, 1))
        figureWindow = plt.figure(figsize=figsize)
        figureWindow.suptitle(summaryTitle, fontsize=20)
        for i in range(len(images)):
            sp = figureWindow.add_subplot(rows, len(images) // rows, i + 1)
            sp.axis('Off')
            if titles is not None:
                sp.set_title(titles[i], fontsize=16)
            plt.imshow(images[i], interpolation=None if interp else 'none')

    @staticmethod
    def __confidenceSortKey(testResultSummary: TestResultSummary):
        return testResultSummary.getPredictionSummary().getTopPrediction().getConfidence()
