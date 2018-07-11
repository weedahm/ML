import csv2data
import numpy as np
#from common.functions import unsupervisedFuncs
from layer_network_tf import ThreeLayerNet

'''
# unsupervised learning
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

# deep learning
trainX = csv2data.get_data('patients_trainingX.csv')
trainY = csv2data.get_data('patients_trainingY.csv')
testX = csv2data.get_data('patients_testX.csv')
testY = csv2data.get_data('patients_testY.csv')
#print(trainX.shape, trainY.shape, testX.shape, testY.shape)
spv_learn = ThreeLayerNet(trainX, trainY, testX, testY)
spv_learn.Net()