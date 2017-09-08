from common.model.deeplearning.test.TestResultSummary import TestResultSummary
import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


# TODO:  Add visualizations for random correct and random incorrect
# TODO:  Add visualization for dead neurons?

class ImagePerformanceVisualizer:
    @staticmethod
    def do_visualizations(test_result_summaries: [TestResultSummary], class_name: str, num_tests_to_visualize: int, visualize_success: bool,
                          visualize_failure: bool, visualize_least_confident: bool, visualize_confusion_matrix: bool):
        if visualize_success:
            ImagePerformanceVisualizer.__visualize_most_confident_successful_predictions(test_result_summaries, class_name, num_tests_to_visualize)

        if visualize_failure:
            ImagePerformanceVisualizer.__visualize_most_confident_failing_predictions(test_result_summaries, class_name, num_tests_to_visualize)

        if visualize_least_confident:
            ImagePerformanceVisualizer.__visualize_least_confidentPredictions(test_result_summaries, class_name, num_tests_to_visualize)

        if visualize_confusion_matrix:
            ImagePerformanceVisualizer.__visualize_confusion_matrix(test_result_summaries)

        plt.show()

    @staticmethod
    def __visualize_most_confident_successful_predictions(test_result_summaries: [TestResultSummary], class_name: str, num_tests_to_visualize: int):
        if len(test_result_summaries) == 0 or num_tests_to_visualize < 1:
            return

        correct = [test_result_summary for test_result_summary in test_result_summaries if (test_result_summary.is_correct() and
                                                                                            test_result_summary.get_actual_class_name() == class_name)]

        sorted_correct = sorted(correct, key=ImagePerformanceVisualizer.__confidence_sort_key, reverse=True)

        to_display = sorted_correct[:num_tests_to_visualize]
        ImagePerformanceVisualizer.__prepare_visualizations(to_display, 'Most Confident Successful Predictions')

    @staticmethod
    def __visualize_most_confident_failing_predictions(test_result_summaries: [TestResultSummary], class_name: str, num_tests_to_visualize: int):
        if len(test_result_summaries) == 0 or num_tests_to_visualize < 1:
            return

        failing = [test_result_summary for test_result_summary in test_result_summaries if (not test_result_summary.is_correct() and
                                                                                            test_result_summary.get_actual_class_name() == class_name)]

        sorted_failing = sorted(failing, key=ImagePerformanceVisualizer.__confidence_sort_key, reverse=True)

        to_display = sorted_failing[:num_tests_to_visualize]
        ImagePerformanceVisualizer.__prepare_visualizations(to_display, 'Most Confident Failing Predictions')

    @staticmethod
    def __visualize_least_confidentPredictions(test_result_summaries: [TestResultSummary], class_name: str, num_tests_to_visualize: int):
        if len(test_result_summaries) == 0 or num_tests_to_visualize < 1:
            return

        test_results_of_class = [test_result_summary for test_result_summary in test_result_summaries if test_result_summary.get_actual_class_name() == class_name]

        sorted_results = sorted(test_results_of_class, key=ImagePerformanceVisualizer.__confidence_sort_key, reverse=False)
        to_display = sorted_results[:num_tests_to_visualize]
        ImagePerformanceVisualizer.__prepare_visualizations(to_display, 'Least Confident Predictions')

    @staticmethod
    def __visualize_confusion_matrix(test_result_summaries: [TestResultSummary]):
        if len(test_result_summaries) == 0:
            return

        actual_classes = []
        predicted_classes = []
        classes_set = set()

        for test_result_summary in test_result_summaries:
            actual_classes.append(test_result_summary.get_actual_class_name())
            predicted_classes.append(test_result_summary.get_prediction_summary().get_top_prediction().get_class_name())
            classes_set.add(test_result_summary.get_actual_class_name())

        sorted_classes = sorted(classes_set)
        confusion_matrix_result = confusion_matrix(actual_classes, predicted_classes)
        ImagePerformanceVisualizer.__prepare_confusion_matrix(confusion_matrix_result, sorted_classes)

    @staticmethod
    def __prepare_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.get_cmap(name="Blues")):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        (This function is copied from the scikit docs.)
        """
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    @staticmethod
    def __prepare_visualizations(test_result_summaries: [TestResultSummary], summary_title: str):
        images = [test_result_summary.get_prediction_summary().get_image_info().get_pil_image() for test_result_summary in test_result_summaries]
        titles = [test_result_summary.get_prediction_summary().get_top_prediction().get_class_name()
                  + ': ' + str(test_result_summary.get_prediction_summary().get_top_prediction().get_confidence()) for test_result_summary in test_result_summaries]
        figsize = (12, 6)
        rows = 1
        interp = False

        plt.interactive(False)
        if len(images) > 0 and type(images[0]) is np.ndarray:
            images = np.array(images).astype(np.uint8)
            if images.shape[-1] != 3:
                images = images.transpose((0, 2, 3, 1))
        figure_window = plt.figure(figsize=figsize)
        figure_window.suptitle(summary_title, fontsize=20)
        for i in range(len(images)):
            sp = figure_window.add_subplot(rows, len(images) // rows, i + 1)
            sp.axis('Off')
            if titles is not None:
                sp.set_title(titles[i], fontsize=16)
            plt.imshow(images[i], interpolation=None if interp else 'none')

    @staticmethod
    def __confidence_sort_key(test_result_summary: TestResultSummary):
        return test_result_summary.get_prediction_summary().get_top_prediction().get_confidence()
