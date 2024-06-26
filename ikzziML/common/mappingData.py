import pandas as pd
import csv
import json
import numpy as np
import os

script_path = os.path.dirname(os.path.abspath(__file__))
DISEASE_ML_INPUT_MAP_PATH = script_path + '/table/ML_input_training_map.json'

def loadCSV(patients_X_csv):
    df_X = pd.read_csv(patients_X_csv, encoding='ms949')
    return df_X

def mappingToTrainingValues(dataframe_X):
    tmp_X = dataframe_X.copy()

    with open(DISEASE_ML_INPUT_MAP_PATH, encoding='utf-8') as data_file:
        val_dic = json.load(data_file)
    
    for feature in val_dic.keys() :
        if feature == 'input-eav' : # 예외key: eav-1 ~ -6 까지
            for i in range(6):
                tmp_X[feature+'-'+str(i+1)] = tmp_X[feature+'-'+str(i+1)].map(val_dic[feature])
        else :
            tmp_X[feature] = tmp_X[feature].map(val_dic[feature])

    tmp_X = tmp_X.fillna(0) # 빈칸 = 0
    tmp_X = tmp_X*1 # True, False => 1, 0
    dataframe_X = tmp_X.copy()
    return dataframe_X

def saveCSV(training_df_X, save_path):
    training_df_X.to_csv(save_path+'X_Training.csv', index=False, encoding='ms949')