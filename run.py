import csv2data
import numpy as np
#from common.functions import unsupervisedFuncs
from common.dataPreprocessing import DataPreprocessing
from layer_network_tf import ThreeLayerNet
import layer_network_tf_inference as lninf

'''
##### unsupervised learning
patients_data = csv2data.get_data('patients_survey_data.csv') # get array data from .csv file 

unspv_learn = unsupervisedFuncs(patients_data)

#unspv_learn.let_PCA(components=3) # run Principal Component Analysis
unspv_learn.let_maniford(method=1, components=30) # run Manifold Learning
#unspv_learn.let_kMC(clusters=3) # run k-Mean Clustering
unspv_learn.let_GMM(clusters=5)

#unspv_learn.show_components_info()

print(unspv_learn.y_data)
#print(unspv_learn.y_prob.round(3))
#print(unspv_learn.y_prob.shape)

#unspv_learn.print_plot() # draw plot
'''

'''
##### deeplearning - training(data preprocessing)
dpp = DataPreprocessing()

trainX = csv2data.get_data('patients_trainingX.csv')
dpp.setMean(trainX)
trainX_ms = dpp.meanSubtraction(trainX)
dpp.setStd(trainX_ms)
trainX_nms = dpp.normalization(trainX_ms)

trainY = csv2data.get_data('patients_trainingY.csv')

testX = csv2data.get_data('patients_testX.csv')
testX_nms = dpp.normalization(dpp.meanSubtraction(testX))

testY = csv2data.get_data('patients_testY.csv')

spv_learn = ThreeLayerNet(trainX_nms, trainY, testX_nms, testY)
spv_learn.Net()
'''

##### deeplearning - training(no data preprocessing)
trainX = csv2data.get_data('patients_trainingX2.csv')
trainY = csv2data.get_data('patients_trainingY2.csv')
testX = csv2data.get_data('patients_testX2.csv')
testY = csv2data.get_data('patients_testY2.csv')
spv_learn = ThreeLayerNet(trainX, trainY, testX, testY)
spv_learn.Net()

'''
##### deeplearning - inference
testX = csv2data.get_data('patients_testX.csv')
lninf.inferenceNet(testX)
'''
