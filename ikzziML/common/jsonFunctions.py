import os
import json
import numpy as np
import pandas as pd
from . import mappingData
from . import manipulateMissingVal

script_path = os.path.dirname(os.path.abspath(__file__))

DISEASE_ML_INPUT_MAP_PATH = script_path + '/table/diseaseMLInput_map_new.json'

def readJsonFile(file_path):
    with open(file_path, encoding='utf-8') as data_file:
        json_data = json.load(data_file)
    return json_data

def castToMLData(bodychart_data):
    with open(DISEASE_ML_INPUT_MAP_PATH, encoding='utf-8') as data_file:
        val_dic = json.load(data_file)

    data = np.zeros((1, len(val_dic)), dtype=object)
    column_list = np.empty(len(val_dic), dtype=object)

    ######## bodychart(json -> ML input Form)
    for i in val_dic.values():
        column_list[i[0]] = i[2]
        if bodychart_data[i[1]][i[2]] != "" : # 공백(값x)의 경우 0
            data[0][i[0]] = bodychart_data[i[1]][i[2]]

    data_df = pd.DataFrame(data[0], index=column_list).T
    ML_data_df = mappingData.mappingToTrainingValues(data_df)
    ML_data_df = manipulateMissingVal.totalMani(ML_data_df)
    data[0] = list(map(float, ML_data_df.values[0]))

    return data
