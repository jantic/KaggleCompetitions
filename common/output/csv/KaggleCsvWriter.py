import pandas as pd

from common.model.deeplearning.prediction.PredictionsSummary import PredictionsSummary


class KaggleCsvWriter:
    @staticmethod
    def writePredictionsForClassIdToCsv(predictionsSummaries: [PredictionsSummary], classId):
        if len(predictionsSummaries) == 0:
            return

        records = []

        for predictionSummary in predictionsSummaries:
            testId = predictionSummary.getTestId()
            confidence = predictionSummary.getConfidenceForClassId(classId)
            clippedConfidence = 0.02 if confidence < 0.02 else (0.98 if confidence > 0.98 else confidence)
            records.append({'id': testId, 'label': clippedConfidence})

        df = pd.DataFrame.from_records(records)
        df['id'] = pd.to_numeric(df['id'])
        df = df.sort_values('id')
        df.to_csv('submission.csv', index=False)
