import csv2data
from common.functions import unsupervisedFuncs

patients_data = csv2data.get_data('patients_survey_data_minimum.csv') # get array data from .csv file 

unspv_learn = unsupervisedFuncs(patients_data)

#unspv_learn.let_PCA(components=3) # run Principal Component Analysis
unspv_learn.let_maniford(method=1, components=50) # run Manifold Learning
#unspv_learn.let_kMC(clusters=3) # run k-Mean Clustering
unspv_learn.let_GMM(clusters=5)

print(unspv_learn.y_data)
#print(unspv_learn.y_prob.round(3))
#print(unspv_learn.y_prob.shape)

unspv_learn.print_plot() # draw plot