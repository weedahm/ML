import json
import sqlite3
import pandas as pd
import csv
import numpy as np
import os

script_path = os.path.dirname(os.path.abspath(__file__))
DISEASE_ML_INPUT_MAP_PATH = script_path + '/table/diseaseMLInput_map_new.json'
DISEASE_ML_OUTPUT_MAP_PATH = script_path + '/table/diseaseMLOutput_map.json'

INPUT_KEYS = ['basic_info', 'bodychart', 'abdominal', 'eav'] # 증상, 신체정보: X
OUTPUT_KEY = ['prescription'] # 약재: Y
MAX_N_SET = 3 # 처방 세트 수

def readDB(file_path):
    con = sqlite3.connect(file_path)
    df_patients = pd.read_sql("SELECT json_data FROM data_collecting_patient", con)

    return df_patients

def castToMLData(df_patients):
    series_data = df_patients['json_data']
    n_rows = series_data.shape[0] # 총 행 수

    with open(DISEASE_ML_INPUT_MAP_PATH, encoding='utf-8') as data_file:
        val_dic_input = json.load(data_file)
    with open(DISEASE_ML_OUTPUT_MAP_PATH, encoding='utf-8') as data_file:
        val_dic_output = json.load(data_file)

    ##### column 명 순서대로 정렬한 list
    output_range = len(val_dic_output)
    column_list_in = np.empty(len(val_dic_input), dtype=object)
    column_list_out = np.empty(output_range*MAX_N_SET, dtype=object)
    column_list_out_set = np.empty(MAX_N_SET, dtype=object)

    for i in val_dic_input.values() :
        column_list_in[i[0]] = i[2]

    for n in range(MAX_N_SET) :
        column_list_out_set[n] = '처방'+str(n+1)
        for i in val_dic_output.keys() :
            column_list_out[val_dic_output[i]+output_range*n] = column_list_out_set[n]+'_'+i

    ##### 학습용 DataFrame 생성
    ML_input_list = np.empty((n_rows, column_list_in.shape[0]), dtype=object)
    ML_output_list = np.zeros((n_rows, column_list_out.shape[0]))
    oneHot_set_list = np.zeros((n_rows, MAX_N_SET), dtype=int)

    i = 0
    for val in series_data :
        json_data = json.loads(val)
        js_input = pd.Series(index=column_list_in)
        js_output = pd.Series(index=column_list_out)

        # 증상, 신체정보 input / 약재 output
        for k in INPUT_KEYS :
            js_input.update(pd.Series(json_data[k]))
        for k in OUTPUT_KEY :
            js_output.update(pd.Series(json_data[k]))
        js_output = js_output.fillna(0)

        ML_input_list[i] = js_input.values
        ML_output_list[i] = js_output.values
        
        # set 수 output
        list_output = list(map(float, js_output.values))
        n_set = 0
        for n in range(MAX_N_SET) :
            if sum(list_output[output_range*n:output_range*(n+1)]) == 0:
                n_set = n
                break
            elif n == (MAX_N_SET-1):
                n_set = n+1

        if n_set != 0:
            oneHot_set_list[i][n_set-1] = 1

        i += 1
    
    df_input = pd.DataFrame(ML_input_list, columns=column_list_in)
    df_output = pd.DataFrame(ML_output_list, columns=column_list_out)
    df_output_set = pd.DataFrame(oneHot_set_list, columns=column_list_out_set)

    return df_input, df_output, df_output_set

def saveToCSV(patients_X_df, patients_Y_df, patients_Y_set_df):
    patients_X_df.to_csv('created_csv/patients_X_new.csv', index=False, encoding='ms949')
    patients_Y_df.to_csv('created_csv/patients_Y_new.csv', index=False, encoding='ms949')
    patients_Y_set_df.to_csv('created_csv/patients_Y_set_new.csv', index=False, encoding='ms949')
