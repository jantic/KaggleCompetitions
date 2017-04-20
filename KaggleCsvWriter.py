import PredictionInfo
import PredictionsSummary
import pandas as pd

class KaggleCsvWriter:
    def writePredictionsForClassIdToCsv(self, predictionsSummaries : [], classId):
        if(len(predictionsSummaries) == 0):
            return

        records = []

        for predictionSummary in predictionsSummaries:
            testId = predictionSummary.getTestId()
            confidence = predictionSummary.getConfidenceForClassId(classId)
            records.append({'id': testId, 'label': confidence})

        df = pd.DataFrame.from_records(records)
        df['id'] = pd.to_numeric(df['id'])
        df = df.sort_values('id')
        df.to_csv('submission.csv', index=False)