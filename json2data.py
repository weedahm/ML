import json
import numpy as np

val_dic = {
    "신장" : [0, 'basic_info', 'input-height'],
    "체중" : [1, 'basic_info', 'input-weight'],
    "복진상" : [2, 'abdominal', '복부상'],
    "복진하좌" : [3, 'abdominal', '복부하'],
    "복진하우" : [4, 'abdominal', '복부하'],
    "두통" : [5, 'bodychart', '두통'],
    "어지러움" : [6, 'bodychart', '현훈'],
    "구취" : [7, 'bodychart', '구취'],
    "소화불량" : [8, 'bodychart', '소화불량'],
    "명치답답" : [9, 'bodychart', '명치답답'],
    "명치통증" : [10, 'bodychart', '명치통증'],
    "체함" : [11, 'bodychart', '체함'],
    "더부룩상" : [12, 'bodychart', '더부룩상'],
    "더부룩하" : [13, 'bodychart', '더부룩하'],
    "트림공복" : [14, 'bodychart', '트림공복'],
    "트림식후" : [15, 'bodychart', '트림식후'],
    "속쓰림공복" : [16, 'bodychart', '속쓰림공복'],
    "속쓰림식후" : [17, 'bodychart', '속쓰림식후'],
    "역류공복" : [18, 'bodychart', '역류공복'],
    "역류식후" : [19, 'bodychart', '역류식후'],
    "오심" : [20, 'bodychart', '오심'],
    "복통" : [21, 'bodychart', '복통'],
    "피로감" : [22, 'bodychart', '피로감'],
    "건망증" : [23, 'bodychart', '건망증'],
    "안구건조" : [24, 'bodychart', '안구건조'],
    "불안감" : [25, 'bodychart', '불안'],
    "두근거림" : [26, 'bodychart', '두근거림'],
    "가슴답답" : [27, 'bodychart', '가슴답답'],
    "가슴통증" : [28, 'bodychart', '가슴통증'],
    "목이물감" : [29, 'bodychart', '목이물감'],
    "등뻐근" : [30, 'bodychart', '등뻐근'],
    "상열감" : [31, 'bodychart', '상열감'],
    "숨참" : [32, 'bodychart', '숨참'],
    "항강" : [33, 'bodychart', '항강'],
    "견통" : [34, 'bodychart', '견통'],
    "단단변" : [35, 'bodychart', '단단'],
    "무른변" : [36, 'bodychart', '무름'],
    "설사" : [37, 'bodychart', '설사'],
    "잔변감" : [38, 'bodychart', '잔변감'],
    "냉대하" : [39, 'bodychart', '냉대하'],
    "구건" : [40, 'bodychart', '구건'],
    "구고" : [41, 'bodychart', '구고']
}

def loadJson(file_path):
    with open(file_path, encoding='utf-8') as data_file:
        json_data = json.load(data_file)
    return json_data

def setData(json_data, num_features = 42):
    data = np.zeros((1, num_features))

    for i in val_dic.values():
        if json_data[i[1]][i[2]] and type(json_data[i[1]][i[2]]) == bool:
            data[0][i[0]] = 6
        elif not json_data[i[1]][i[2]]: # or (json_data[i[1]][i[2]] == ""):
            data[0][i[0]] = 0
        else:
            data[0][i[0]] = json_data[i[1]][i[2]]

    print(data)
    return data
