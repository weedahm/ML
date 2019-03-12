import numpy as np
import pandas as pd
import json
import os

script_path = os.path.dirname(os.path.abspath(__file__))
ML_INPUT_MISSING_MAP_PATH = script_path + '/table/ML_input_missing_val_map.json'
MISSING_ABDO_VALUE = 99.4

def maniAbdominal(data):
    i = 0
    for row in data:
        if all(row) != True: # missing row
            if any(row) == False: # 0,0,0
                data[i] = MISSING_ABDO_VALUE
            else: # 하나라도 값이 있는 경우
                data[i][data[i]==0] = max(row)
                #data[i] = [max(row) if x==0 else x for x in row]
        i += 1
    return data

def totalMani(df_train_X):
    with open(ML_INPUT_MISSING_MAP_PATH, encoding='utf-8') as data_file:
        val_dic = json.load(data_file)

    train_X_patial_list = df_train_X[val_dic['abdominal']].values.astype(np.float)
    train_X_patial_list = maniAbdominal(train_X_patial_list)
    
    train_X_patial_df = pd.DataFrame(train_X_patial_list, columns=val_dic['abdominal'])
    df_train_X.update(train_X_patial_df)

    return df_train_X

def deleteZeroPres(df_train_X, df_Y, df_Y_set, save_path):
    zeroPres_indexList = df_Y_set.loc[df_Y_set.sum(axis=1) == 0].index.values
    
    df_train_X = df_train_X.drop(index=zeroPres_indexList)
    df_Y = df_Y.drop(index=zeroPres_indexList)
    df_Y_set = df_Y_set.drop(index=zeroPres_indexList)
    
    df_train_X.to_csv(save_path+'X_Training.csv', index=False, encoding='ms949')
    df_Y.to_csv(save_path+'Y.csv', index=False, encoding='ms949')
    df_Y_set.to_csv(save_path+'Y_set.csv', index=False, encoding='ms949')
