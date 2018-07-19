import numpy as np
'''
LOW_DAM = 11.4
LOW_SO = -0.1
LOW_SIN = 0.8
LOW_SOON = 5.7
LOW_GI = 0.3
LOW_TUK = 4
'''
def setScore(predict_data):
    tmp_score = predict_data
    tmp_score[0][0] = tmp_score[0][0] * 2
    tmp_score[0][1] = tmp_score[0][1] * 10
    tmp_score[0][2] = tmp_score[0][2] * 10
    tmp_score[0][3] = tmp_score[0][3] * 4.5
    tmp_score[0][4] = tmp_score[0][4] * 20
    tmp_score[0][5] = tmp_score[0][5] * 5

    for i in range(6):
        if tmp_score[0][i] > 100:
            tmp_score[0][i] = 100
        elif tmp_score[0][i] < 0:
            tmp_score[0][i] = 0
    
    score = tmp_score.round(1)
    return score
