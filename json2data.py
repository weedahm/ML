import json
import numpy as np

with open('data/data.json') as data_file:
    json_data = json.load(data_file)

val_dic = {
    "HEIGHT" : [0, 'basic_info', 'input-height'],
    "WEIGHT" : [1, 'basic_info', 'input-weight'],
    "복진상" : [2, 'bodychart', '복진-상'], # 복진 따로 이동 예정
    "복진하좌" : [3, 'bodychart', '복진-하'],
    "복진하우" : [4, 'bodychart', '복진-하'],
    "두통" : [5, 'bodychart', '두통'],
    "어지러움" : [6, 'bodychart', '현훈'],
    "구취" : [7, 'bodychart', '구취'],
    "소화불량" : [8, 'bodychart', '소화불량'],
    "명치답답공복" : [9, 'bodychart', '명치답답'], # 하나로 합칠것임
    "명치답답식후" : [10, 'bodychart', '명치답답'],
    "명치답답항상" : [11, 'bodychart', '명치답답'],
    "명치통증" : [12, 'bodychart', '명치통증'], # (하나 만들기)
    "잘체함" : [13, 'bodychart', '체함'],
    "더부룩상" : [14, 'bodychart', '더부룩'],
    "더부룩하" : [15, 'bodychart', '더부룩'],
    "트림공복" : [16, 'bodychart', '트림'],
    "트림식후" : [17, 'bodychart', '트림'],
    "속쓰림공복" : [18, 'bodychart', '속쓰림'],
    "속쓰림식후" : [19, 'bodychart', '속쓰림'],
    "역류공복" : [20, 'bodychart', '역류'],
    "역류식후" : [21, 'bodychart', '역류'],
    "메스꺼움" : [22, 'bodychart', '오심'],
    "복통" : [23, 'bodychart', '복통'],
    "피로감" : [24, 'bodychart', '피로감'],
    "건망증" : [25, 'bodychart', '건망증'],
    "안구통증" : [26, 'bodychart', '안구통증'], # 없음
    "안구건조" : [27, 'bodychart', '안구건조'],
    "불안감" : [28, 'bodychart', '불안'],
    "가슴이두근거림" : [29, 'bodychart', '심계'],
    "가슴" : [30, 'bodychart', '흉민'], # 가슴흉민, 가슴흉통 나눠야함
    "목이물감" : [31, 'bodychart', '목이물감'],
    "등뻐근" : [32, 'bodychart', '등뻐근'], # (하나 만들기)
    "뇌로열이오름" : [33, 'bodychart', '상열감'],
    "숨참" : [34, 'bodychart', '숨참'],
    "피부이상" : [35, 'bodychart', '피부이상'], # 없애기
    "담결림" : [36, 'bodychart', '담결림'], # 없애기
    "뒷목뻣뻣" : [37, 'bodychart', '항강'], # (하나 만들기, 견통 밑에 [항강])
    "어깨결림" : [38, 'bodychart', '견통'],
    "변비" : [39, 'bodychart', '경변'],
    "설사" : [40, 'bodychart', '설사'],
    "잔변감" : [41, 'bodychart', '잔변감'], # 없애기
    "냉대하" : [42, 'bodychart', '냉대하'],
    "구건" : [43, 'bodychart', '구건'],
    "구고" : [44, 'bodychart', '구고']
}

def setData(num_features = 45):
        data = np.zeros(num_features)

        for i in val_dic.values():
            if i[0] == 0 or i[0] == 1:
                data[i[0]] = json_data[i[1]][i[2]]

        print(data)
        print(json_data)

        # reshape 해야함 axis 1 에 값 들어가게
        
setData()
