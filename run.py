from ikzziML import learningFunction

def menu():
    print("1. Training: Unsupervised")
    print("2. Training: Supervised - Set")
    print("3. Training: Supervised - Prescription")
    print("4. Inferencing: Set & Prescription")
    try:
        m = int(input("Select: "))
    except ValueError:
        print("Invalid number")
    return m

m = menu()

############## 1.   INPUT & SETTING DATA PATH    #################
unSupervised_data_path = 'data/csv/new/patients_Y.csv'
X_path = 'data/csv/new/patients_X_Training.csv'
Y_path_pre = 'data/csv/new/patients_Y.csv'
Y_path_set = 'data/csv/new/patients_Y_set.csv'

inference_testX_path = 'data/json/test4.json'

############## 2.   TRAINING or INFERENCING    #################
if(m == 1):
    learningFunction.unsupervised_learning(unSupervised_data_path, dimension_reduction=2, clustering=2, n_component=2, n_cluster=5)
elif(m == 2):
    learningFunction.supervised_learning_training(X_path, Y_path_pre, Y_path_set, datapps=1, isSet=True)
elif(m == 3):
    learningFunction.supervised_learning_training(X_path, Y_path_pre, Y_path_set, datapps=1, isSet=False)
else: # m == 4
    inference_textX = learningFunction.readBodychart(inference_testX_path)

    # set 수
    predict_n_set = learningFunction.supervised_learning_inference(inference_textX, isSet=True)
    print("Predict", predict_n_set, "Prescription SETs.")

    # 약재 중량(LIST)
    predict_data = learningFunction.supervised_learning_inference(inference_textX, isSet=False, infer_n_set=predict_n_set)

    # 약재 중량(DIC)
    predict_data_dic = learningFunction.dataToDic(predict_data, n_set=predict_n_set)

    # 그룹 점수(DIC)
    score = learningFunction.groupScore(predict_data_dic)

    # 최종 데이터(DIC)
    data = learningFunction.totalDic(score, predict_data_dic)
    print(data)
    print(sum(data['SET']['SET1'].values())) # 총 중량 확인
    print(sum(data['SET']['SET2'].values()))
    print(sum(data['SET']['SET3'].values()))