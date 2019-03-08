import os
import numpy as np
import json

script_path = os.path.dirname(os.path.abspath(__file__))

DISEASE_ML_OUTPUT_NAME_PATH = script_path + '/table/diseaseMLOutput_map.json'
DISEASE_GROUP_SCORE_MAP_PATH = script_path + '/table/diseaseGroup_score_map.json'
DISEASE_GROUP_SCORE_CALC_PATH = script_path + '/table/diseaseGroup_score_calc.json'

def totalDic(sco_dic, set_dic):
    total_dic = {'SCORE':sco_dic, 'SET':set_dic}
    return total_dic

def setToDic(predict_data, n_set):
    with open(DISEASE_ML_OUTPUT_NAME_PATH, encoding='utf-8') as data_file:
        val_dic = json.load(data_file)

    data = { 'SET1':{}, 'SET2':{}, 'SET3':{} }

    output_range = len(val_dic)

    for i in val_dic.keys():
        data['SET1'][i] = predict_data[0][val_dic[i]]
        data['SET2'][i] = predict_data[0][val_dic[i]+output_range]
        data['SET3'][i] = predict_data[0][val_dic[i]+(output_range*2)]

    if n_set == 1:
        del data['SET2']
        del data['SET3']
    elif n_set == 2:
        del data['SET3']
    else:
        pass
    
    return data

def calcScore(sum_dic_data):
    with open(DISEASE_GROUP_SCORE_MAP_PATH, encoding='utf-8') as data_file:
        val_dic = json.load(data_file)

    with open(DISEASE_GROUP_SCORE_CALC_PATH, encoding='utf-8') as data_file:
        score_dic = json.load(data_file)

    ##### sum to 그룹 약재 중량
    group_score = {}
    for key in val_dic.keys():
        a = 0
        for i in val_dic[key]:
            a = a + sum_dic_data[i]
        group_score[key] = a

    ##### 약재 중량 -> Score (0 ~ 100 으로 stretch)
    for key in group_score.keys():
        group_score[key] = group_score[key] * score_dic[key]

        if group_score[key] > 100:
            group_score[key] = 100
        elif group_score[key] < 0:
            group_score = 0
        else:
            group_score[key] = group_score[key].round(0)
    return group_score

def sumOneSet(dic_data):
    large_keyset = list(dic_data.keys())
    large_key = large_keyset[0]
    oneSet = {}
    
    for i in dic_data[large_key].keys():
        a = 0
        for key in dic_data.keys():
            a = a + dic_data[key][i]
        oneSet[i] = a

    return oneSet
