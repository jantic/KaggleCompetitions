from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary
import pathlib2
from common.utils import utils
from collections import OrderedDict
import pandas as pd
import numpy as np

class CsvSubmissionWritter:
    @staticmethod
    def write_predictions_to_csv(pred_summaries: [PredictionsSummary]):
        if len(pred_summaries) == 0:
            return

        records = []

        for pred_summary in pred_summaries:
            image_name = pathlib2.PureWindowsPath(pred_summary.get_image_info().get_image_path()).name

            predictions = pred_summary.get_all_predictions()
            raw_confidence_array = np.zeros(10)

            for prediction in predictions:
                confidence = prediction.get_confidence()
                raw_confidence_array[prediction.get_class_id()] = confidence

            clipped_confidence = utils.do_clip(raw_confidence_array, 0.97)

            row = OrderedDict([('img', image_name), ('c0', clipped_confidence[0]), ('c1', clipped_confidence[1]), ('c2', clipped_confidence[2]),
                               ('c3', clipped_confidence[3]), ('c4', clipped_confidence[4]), ('c5', clipped_confidence[5]), ('c6', clipped_confidence[6]),
                               ('c7', clipped_confidence[7]), ('c8', clipped_confidence[8]), ('c9', clipped_confidence[9])])

            records.append(row)

        df = pd.DataFrame.from_records(records)
        df = df.sort_values('img')
        df.to_csv('submission.csv', index=False)
