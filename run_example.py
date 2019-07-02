from ikzziML import learningFunction

# set 수
predict_n_set = learningFunction.supervised_learning_inference(inference_textX, isSet=True)

# 약재 중량(LIST)
predict_data = learningFunction.supervised_learning_inference(inference_textX, isSet=False, infer_n_set=predict_n_set)

# 약재 중량(DIC)
predict_data_dic = learningFunction.dataToDic(predict_data, n_set=predict_n_set)

# 그룹 점수(DIC)
score = learningFunction.groupScore(predict_data_dic)

# 최종 데이터(DIC)
data = learningFunction.totalDic(score, predict_data_dic)
print(data)
