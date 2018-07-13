import csv2data
import numpy as np
#from common.functions import unsupervisedFuncs
from common.dataPreprocessing import DataPreprocessing
from layer_network_tf import ThreeLayerNet
import layer_network_tf_inference as lninf

'''
def unsupervised_learning():
    ##### unsupervised learning
    patients_data = csv2data.get_data('patients_survey_data.csv') # get array data from .csv file 

    unspv_learn = unsupervisedFuncs(patients_data)

    #unspv_learn.let_PCA(components=3) # run Principal Component Analysis
    unspv_learn.let_maniford(method=1, components=30) # run Manifold Learning
    #unspv_learn.let_kMC(clusters=3) # run k-Mean Clustering
    unspv_learn.let_GMM(clusters=5) # run Gaussian Mixture Models

    #unspv_learn.show_components_info() # draw plot about information loss of PCA

    print(unspv_learn.y_data)
    #print(unspv_learn.y_prob.round(3))
    #print(unspv_learn.y_prob.shape)

    unspv_learn.print_plot() # draw plot
'''

def supervised_learning_training(datapps = 0):
    ##### deeplearning - training(data preprocessing - standardization)
    if datapps == 1:
        dpp = DataPreprocessing()

        trainX = csv2data.get_data('data/csv/patients_trainingX.csv')
        trainY = csv2data.get_data('data/csv/patients_trainingY.csv')
        testX = csv2data.get_data('data/csv/patients_testX.csv')
        testY = csv2data.get_data('data/csv/patients_testY.csv')

        dpp.setMeanStd(trainX)

        trainX_std = dpp.standardization(trainX)
        testX_std = dpp.standardization(testX)

        print(trainX_std)

        spv_learn = ThreeLayerNet(trainX_std, trainY, testX_std, testY)
        spv_learn.Net()

    ##### deeplearning - training(data preprocessing - minmax scaler)
    elif datapps == 2:
        dpp = DataPreprocessing()
        dppY = DataPreprocessing()

        trainX = csv2data.get_data('data/csv/patients_trainingX4_wh.csv')
        trainY = csv2data.get_data('data/csv/patients_trainingY4.csv')
        testX = csv2data.get_data('data/csv/patients_testX4_wh.csv')
        testY = csv2data.get_data('data/csv/patients_testY4.csv')

        print(trainX.shape, trainY.shape, testX.shape, testY.shape)

        dpp.setMinDistance(trainX)
        dppY.setMinDistance(trainY)

        trainX_mms = dpp.minMaxScaler(trainX)
        testX_mms = dpp.minMaxScaler(testX)

        #print(trainX_mms)

        spv_learn = ThreeLayerNet(trainX_mms, trainY, testX_mms, testY)
        spv_learn.Net()

    else:
        ##### deeplearning - training(no data preprocessing)
        trainX = csv2data.get_data('data/csv/patients_trainingX4.csv')
        trainY = csv2data.get_data('data/csv/patients_trainingY4_one.csv')
        testX = csv2data.get_data('data/csv/patients_testX4.csv')
        testY = csv2data.get_data('data/csv/patients_testY4_one.csv')

        print(trainX.shape, trainY.shape, testX.shape, testY.shape)

        spv_learn = ThreeLayerNet(trainX, trainY, testX, testY)
        spv_learn.Net()

def supervised_learning_inference():
    ##### deeplearning - inference
    testX = csv2data.get_data('patients_testX.csv')
    lninf.inferenceNet(testX)

#unsupervised_learning()

supervised_learning_training(datapps=2)

#supervised_learning_inference()
