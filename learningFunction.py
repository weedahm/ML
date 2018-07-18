import csv2data
import numpy as np
import csv
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

def supervised_learning_training(trainX_path, trainY_path, testX_path, testY_path, datapps = 0):
    trainX = csv2data.get_data(trainX_path)
    trainY = csv2data.get_data(trainY_path)
    testX = csv2data.get_data(testX_path)
    testY = csv2data.get_data(testY_path)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    dpp = DataPreprocessing()

    f = open('patients_2layerNN/dataPreprocessing.csv', 'w', newline='')
    wr = csv.writer(f)

    ##### deeplearning - training(data preprocessing - standardization)
    if datapps == 1:
        dpp.setMeanStd(trainX)
        trainX = dpp.standardization(trainX)
        testX = dpp.standardization(testX)
        wr.writerow('1')
        wr.writerow(dpp.mean)
        wr.writerow(dpp.std)
    ##### deeplearning - training(data preprocessing - minmax scaler)
    elif datapps == 2:
        dpp.setMinDistance(trainX)
        trainX = dpp.minMaxScaler(trainX)
        testX = dpp.minMaxScaler(testX)
        wr.writerow('2')
        wr.writerow(dpp.min)
        wr.writerow(dpp.distance)
    ##### deeplearning - training(no data preprocessing)
    else:
        wr.writerow('0')
    f.close()

    spv_learn = ThreeLayerNet(trainX, trainY, testX, testY)
    spv_learn.Net()

def supervised_learning_inference(testX_path):
    ##### deeplearning - inference
    testX = csv2data.get_data(testX_path)
   
    f = open('patients_2layerNN/dataPreprocessing.csv', 'r')
    rdr = csv.reader(f)
    data = []
    for line in rdr:
        data.append(line)
    f.close()
    datapps = (np.array(data[0])).astype(np.float)
    dpp = DataPreprocessing()
    
    if datapps == 1:
        dpp.mean = (np.array(data[1])).astype(np.float)
        dpp.std = (np.array(data[2])).astype(np.float)
        testX_pps = dpp.standardization(testX)
    
    elif datapps == 2:
        dpp.min = (np.array(data[1])).astype(np.float)
        dpp.distance = (np.array(data[2])).astype(np.float)
        testX_pps = dpp.minMaxScaler(testX)

    else:
        testX_pps = testX

    lninf.inferenceNet(testX_pps)
