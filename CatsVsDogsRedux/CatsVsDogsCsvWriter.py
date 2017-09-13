import pandas as pd

from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary


class CatsVsDogsCsvWriter:
    @staticmethod
    def write_predictions_for_class_id_to_csv(prediction_summaries: [PredictionsSummary], class_id):
        if len(prediction_summaries) == 0:
            return

        records = []

        for prediction_summary in prediction_summaries:
            test_id = prediction_summary.get_test_id()
            confidence = prediction_summary.get_confidence_for_class_id(class_id)
            clipped_confidence = 0.01 if confidence < 0.01 else (0.99 if confidence > 0.99 else confidence)
            records.append({'id': test_id, 'label': clipped_confidence})

        df = pd.DataFrame.from_records(records)
        df['id'] = pd.to_numeric(df['id'])
        df = df.sort_values('id')
        df.to_csv('submission.csv', index=False)
