import csv2data
from common import functions

patients_data = csv2data.get_data('patients_survey_data_minimum.csv')
functions.let_PCA(patients_data)
functions.show_components_info(patients_data)