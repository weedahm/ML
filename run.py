from ikzziML import learningFunction

############## 1.   INPUT & SETTING DATA PATH    #################
#unSupervised_data_path = 'data/csv/new/patients_Y_new.csv'
X_path = 'data/csv/new/patients_X_Training.csv'
Y_path = 'data/csv/new/patients_Y.csv'

inference_testX_path = 'data/json/test.json'

############## 2.   SELECT TRAINING or INFERENCING    #################
#learningFunction.unsupervised_learning(unSupervised_data_path, dimension_reduction=2, clustering=2, n_component=2, n_cluster=5)
#learningFunction.supervised_learning_training(X_path, Y_path, datapps=1, isSet=True)
#learningFunction.supervised_learning_training(X_path, Y_path, datapps=1, isSet=False, n_set=2)

inference_textX = learningFunction.readBodychart(inference_testX_path)
'''
# set 수
predict_n_set = learningFunction.supervised_learning_inference(inference_textX, isSet=True)
print("Predict", predict_n_set, "Prescription SETs.")

# 약재 중량(LIST)
predict_data = learningFunction.supervised_learning_inference(inference_textX, isSet=False, n_set=predict_n_set)

# 약재 중량(DIC)
predict_data_dic = learningFunction.dataToDic(predict_data, n_set=predict_n_set)

# 그룹 점수(DIC)
score = learningFunction.groupScore(predict_data_dic)

# 최종 데이터(DIC)
data = learningFunction.totalDic(score, predict_data_dic)
print(data)
print(sum(data['SET']['SET1'].values()), sum(data['SET']['SET2'].values())) # 총 중량 확인
'''