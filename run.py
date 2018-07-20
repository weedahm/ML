import learningFunction as lf
import common.calcFunctions as cf

#lf.unsupervised_learning()

trainX_path = 'data/csv/patients_trainingX6.csv'
trainY_path = 'data/csv/patients_trainingY6.csv'
#testX_path = 'data/csv/patients_testX6.csv'
testX_path = 'data/data.json'
testY_path = 'data/csv/patients_testY6.csv'
#lf.supervised_learning_training(trainX_path, trainY_path, testX_path, testY_path, datapps=2)
predict_data = lf.supervised_learning_inference(testX_path, isTraining=False)
print(predict_data)
predict_score = cf.setScore(predict_data)
print(predict_score)
