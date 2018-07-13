import numpy as np

class DataPreprocessing:
    def __init__(self):
        self.mean = []
        self.std = []
        self.min = []

    def setMeanStd(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data-self.mean, axis=0)

    def setMinDistance(self, data):
        self.min = np.min(data, axis=0)
        self.distance = np.max(data - self.min, axis=0)

    def standardization(self, data):
        data_out = (data - self.mean) / self.std
        return data_out

    def minMaxScaler(self, data):
        data_out = (data - self.min) / self.distance
        return data_out