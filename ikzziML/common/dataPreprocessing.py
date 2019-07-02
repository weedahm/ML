import numpy as np

class DataPreprocessing:
    def __init__(self):
        self.mean = []
        self.std = []
        self.min = []
        self.distance = []

    def setMeanStd(self, data):
        self.mean = np.mean(data, axis=0)
        temp_std = np.std(data, axis=0)
        temp_std[temp_std==0] = 1.0
        self.std = temp_std

    def setMinDistance(self, data):
        self.min = np.min(data, axis=0)
        self.distance = np.max(data - self.min, axis=0)

    def standardization(self, data):
        data_out = (data - self.mean) / self.std
        data_out = np.nan_to_num(data_out)
        return data_out

    def minMaxScaler(self, data):
        data_out = (data - self.min) / self.distance
        data_out = np.nan_to_num(data_out)
        return data_out