import os
import json
import numpy as np
import pandas as pd
from . import mappingData

script_path = os.path.dirname(os.path.abspath(__file__))

DISEASE_ML_INPUT_MAP_PATH = script_path + '/table/diseaseMLInput_map_new.json'

def readJsonFile(file_path):
    with open(file_path, encoding='utf-8') as data_file:
        json_data = json.load(data_file)
    return json_data

def castToMLData(bodychart_data):
    with open(DISEASE_ML_INPUT_MAP_PATH, encoding='utf-8') as data_file:
        val_dic = json.load(data_file)

    #data = np.zeros((1, len(val_dic)))
    data = np.zeros((1, len(val_dic)), dtype=object)
    column_list = np.empty(len(val_dic), dtype=object)
    ######## bodychart(json -> ML input Form)
    for i in val_dic.values():
        column_list[i[0]] = i[2]
        data[0][i[0]] = bodychart_data[i[1]][i[2]]
        '''
        if bodychart_data[i[1]][i[2]] and type(bodychart_data[i[1]][i[2]]) == bool: # true
            data[0][i[0]] = 6
        elif not bodychart_data[i[1]][i[2]]: # or (bodychart_data[i[1]][i[2]] == ""): # false / "" / "0" / "없음"
            data[0][i[0]] = 0
        else: # 0을 제외한 수
            data[0][i[0]] = bodychart_data[i[1]][i[2]]
        '''
    data_df = pd.DataFrame(data[0], index=column_list).T
    ML_data_df = mappingData.mappingToTrainingValues(data_df)
    #print(ML_data_df) ---- "" null 값 0으로 대체해야함.
    #print(ML_data_df.values)
    #data[0] = ML_data_df.values

    return data
