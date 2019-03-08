import os
import numpy as np
import csv

from .common import csv2data
from .common import jsonFunctions
from .common import calcFunctions
from .common.dataPreprocessing import DataPreprocessing

script_path = os.path.dirname(os.path.abspath(__file__))

NUMBER_OF_TRAINING_DATA = 600
# 추후 데이터 쌓여서 one, three 네트워크 학습하면 변경. (현재는 모두 two로 사용)
DATA_PREPRO_PATH_ONE = script_path + '/model/patients_2layerNN_two/dataPreprocessing.csv'
DATA_PREPRO_PATH_TWO = script_path + '/model/patients_2layerNN_two/dataPreprocessing.csv'
DATA_PREPRO_PATH_THREE = script_path + '/model/patients_2layerNN_two/dataPreprocessing.csv'
DATA_PREPRO_PATH_SET = script_path + '/model/patients_2layerNN_set/dataPreprocessing.csv'

def datasetSplit(data_set_X, data_set_Y, n_train=600):
    """Split dataset from [full] to [train/test] - Random(uniform) sampling

    Parameters
    ----------
    data_set_X : ndarray, shape (n_samples, n_feartures)
        X is input of learning data

    data_set_Y : ndarray, shape (n_samples, n_feartures)
        Y is output(label) of learning data

    n_train : int, default: 600
        Number of split training samples.

    Returns
    -------
    trainX : ndarray, shape (n_samples, n_feartures)
        input of training data

    trainY : ndarray, shape (n_samples, n_feartures)
        output of training data, equal to trainX's n_samples and have same index order

    testX : ndarray, shape (n_samples, n_feartures)
        input of test data

    testY : ndarray, shape (n_samples, n_feartures)
        output of test data, equal to trainX's n_samples and have same index order
    """
    batch_mask = np.random.choice(data_set_X.shape[0], n_train, replace=False)
    mask = np.zeros(824, dtype=bool)
    mask[batch_mask] = True

    trainX, testX = data_set_X[mask], data_set_X[~mask]
    trainY, testY = data_set_Y[mask], data_set_Y[~mask]
    
    return trainX, trainY, testX, testY

##### unsupervised learning
def unsupervised_learning(data_path, dimension_reduction=0, clustering=0, n_component=3, n_cluster=3):
    from .common.functions import unsupervisedFuncs
    
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

##### supervised learning
def supervised_learning_training(X_path, Y_path, datapps=0, isSet=False, n_set=2):
    from .layer_network_tf import ThreeLayerNet
    from .layer_network_tf_set import TwoLayerNet

    if isSet:
        data_prepro_path = DATA_PREPRO_PATH_SET
    else:
        if n_set == 1:
            data_prepro_path = DATA_PREPRO_PATH_ONE
        elif n_set == 2:
            data_prepro_path = DATA_PREPRO_PATH_TWO
        elif n_set == 3:
            data_prepro_path = DATA_PREPRO_PATH_THREE

    data_X = csv2data.get_data(X_path)
    data_Y = csv2data.get_data(Y_path)
    
    trainX, trainY, testX, testY = datasetSplit(data_X, data_Y, n_train=NUMBER_OF_TRAINING_DATA)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    
    dpp = DataPreprocessing()
    
    write_set = []
    size = [trainX.shape[1], trainY.shape[1]]

    ##### deeplearning - training(data preprocessing - standardization)
    if datapps == 1:
        dpp.setMeanStd(trainX)
        trainX = dpp.standardization(trainX)
        testX = dpp.standardization(testX)
        write_set.append('1')
        write_set.append(dpp.mean)
        write_set.append(dpp.std)
    ##### deeplearning - training(data preprocessing - minmax scaler)
    elif datapps == 2:
        dpp.setMinDistance(trainX)
        trainX = dpp.minMaxScaler(trainX)
        testX = dpp.minMaxScaler(testX)
        write_set.append('2')
        write_set.append(dpp.min)
        write_set.append(dpp.distance)
    ##### deeplearning - training(no data preprocessing)
    else:
        write_set.append('0')

    import os
    os.makedirs(os.path.dirname(data_prepro_path), exist_ok=True)
    csv2data.set_data(data_prepro_path, write_set)

    if isSet:
        spv_learn = TwoLayerNet(trainX, trainY, testX, testY, size)
    else:
        spv_learn = ThreeLayerNet(trainX, trainY, testX, testY, size)

    spv_learn.Net()

##### deeplearning - inference
def supervised_learning_inference(testX_json, isSet=False, n_set=2):
    from . import layer_network_tf_inference as lninf
    from . import layer_network_tf_set_inference as lnsetinf

    if isSet:
        data_prepro_path = DATA_PREPRO_PATH_SET
    else:
        if n_set == 1:
            data_prepro_path = DATA_PREPRO_PATH_ONE
        elif n_set == 2:
            data_prepro_path = DATA_PREPRO_PATH_TWO
        elif n_set == 3:
            data_prepro_path = DATA_PREPRO_PATH_THREE

    f = open(data_prepro_path, 'r')
    rdr = csv.reader(f)
    data = []
    for line in rdr:
        data.append(line)
    f.close()

    datapps = (np.array(data[0])).astype(np.float)
    dpp = DataPreprocessing()

    testX = jsonFunctions.castToMLData(testX_json)

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

    if isSet:
        predict_data = lnsetinf.inferenceNet(testX_pps)
    
    else:
        if n_set == 1:
            predict_data = lninf.inferenceNet(testX_pps) # 처방 약재set 수에 따라 다른 network 사용
        elif n_set == 2:
            predict_data = lninf.inferenceNet(testX_pps)
        elif n_set == 3:
            predict_data = lninf.inferenceNet(testX_pps)
    
    return predict_data

def readBodychart(file_path):
    """Read Bodychart from json file

    Parameters
    ----------
    file_path : String (File path)

    Returns
    -------
    data : dictionary
        Bodychart data, dictionary form.
    """
    return jsonFunctions.readJsonFile(file_path)

def groupScore(dic_data):
    sum_dic_data = calcFunctions.sumOneSet(dic_data)
    return calcFunctions.calcScore(sum_dic_data)

def totalDic(sco_dic, set_dic):
    return calcFunctions.totalDic(sco_dic, set_dic)

def dataToDic(predict_data, n_set):
    return calcFunctions.setToDic(predict_data, n_set)
