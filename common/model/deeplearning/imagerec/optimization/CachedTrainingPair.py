import numpy as np

class CachedTrainingPair:
    def __init__(self, feature_array: np.ndarray, label_array: np.ndarray):
        self.__FEATURE_ARRAY = feature_array
        self.__LABEL_ARRAY = label_array

    def get_feature_array(self):
        return self.__FEATURE_ARRAY

    def get_label_array(self):
        return self.__LABEL_ARRAY