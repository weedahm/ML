import csv2data
from common.functions import unsupervised_funcs

patients_data = csv2data.get_data('patients_survey_data_minimum.csv') # get array data from .csv file 

unspv_learn = unsupervised_funcs(patients_data)

unspv_learn.let_PCA(components=3) # run Principal Component Analysis
unspv_learn.let_kMC(clusters=5) # run k-Mean Clustering
unspv_learn.print_plot() # draw plot