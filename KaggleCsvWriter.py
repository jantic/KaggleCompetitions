import PredictionInfo
import PredictionsSummary
import pandas as pd

class KaggleCsvWriter:
    def writePredictionsToCsv(self, predictionsSummaries : []):
        if(len(predictionsSummaries) == 0):
            return

        records = []

        for predictionSummary in predictionsSummaries:
            records.append({'id': predictionSummary.getTestId(), 'label': predictionSummary.getTopPrediction().getClassId()})

        df = pd.DataFrame.from_records(records)
        df['id'] = pd.to_numeric(df['id'])
        df = df.sort_values('id')
        df.to_csv('submission.csv', index=False)