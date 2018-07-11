import numpy as np

class DataPreprocessing:
    def __init__(self):
        self.mean = .0
        self.std = .0

    def setMean(self, data):
        self.mean = np.mean(data, axis=0)

    def setStd(self, data):
        self.std = np.std(data, axis=0)

    def meanSubtraction(self, data):
        data_out = data - self.mean
        return data_out

    def normalization(self, data):
        data_out = data / self.std
        return data_out
