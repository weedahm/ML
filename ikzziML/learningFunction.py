import csv2data
import numpy as np
import csv
from common.functions import unsupervisedFuncs
from common.dataPreprocessing import DataPreprocessing
from layer_network_tf import ThreeLayerNet
import layer_network_tf_inference as lninf
import json2data

def unsupervised_learning(data_path, dimension_reduction=0, clustering=0, n_component=3, n_cluster=3):
    ##### unsupervised learning
    patients_data = csv2data.get_data(data_path) # get array data from .csv file 

    dpp = DataPreprocessing()
    dpp.setMinDistance(patients_data)
    data = dpp.minMaxScaler(patients_data)
    print(data)
    unspv_learn = unsupervisedFuncs(data)

    if(dimension_reduction == 1): # run Principal Component Analysis
        unspv_learn.let_PCA(components=n_component) 
    elif(dimension_reduction == 2): # run Manifold Learning
        unspv_learn.let_maniford(method=3, components=n_component)

    if(clustering == 1): # run k-Mean Clustering
        unspv_learn.let_kMC(clusters=n_cluster)
    elif(clustering == 2): # run Gaussian Mixture Models
        unspv_learn.let_GMM(clusters=n_cluster)

    #unspv_learn.show_components_info() # draw plot about information loss of PCA

    print(unspv_learn.y_data)
    #print(unspv_learn.y_prob.round(3))
    #print(unspv_learn.y_prob.shape)

    unspv_learn.print_plot() # draw plot


def supervised_learning_training(trainX_path, trainY_path, testX_path, testY_path, datapps = 0):
    trainX = csv2data.get_data(trainX_path)
    trainY = csv2data.get_data(trainY_path)
    testX = csv2data.get_data(testX_path)
    testY = csv2data.get_data(testY_path)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    dpp = DataPreprocessing()

    f = open('patients_2layerNN/dataPreprocessing.csv', 'w', newline='')
    wr = csv.writer(f)
    input_size = trainX.shape[1]
    output_size = trainY.shape[1]
    size = []
    size.append(input_size)
    size.append(output_size)
    wr.writerow(size)
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

    spv_learn = ThreeLayerNet(trainX, trainY, testX, testY, size)
    spv_learn.Net()

def supervised_learning_inference(testX_path):
    ##### deeplearning - inference
    f = open('patients_2layerNN/dataPreprocessing.csv', 'r')
    rdr = csv.reader(f)
    data = []
    for line in rdr:
        data.append(line)
    f.close()
    size = (np.array(data[0])).astype(np.int)
    datapps = (np.array(data[1])).astype(np.float)
    dpp = DataPreprocessing()

    testX = json2data.setData(json2data.loadJson(testX_path), num_features=size[0])

    if datapps == 1:
        dpp.mean = (np.array(data[2])).astype(np.float)
        dpp.std = (np.array(data[3])).astype(np.float)
        testX_pps = dpp.standardization(testX) 
    elif datapps == 2:
        dpp.min = (np.array(data[2])).astype(np.float)
        dpp.distance = (np.array(data[3])).astype(np.float)
        testX_pps = dpp.minMaxScaler(testX)
    else:
        testX_pps = testX

    predict_data = lninf.inferenceNet(testX_pps, size)
    return predict_data
    