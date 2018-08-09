import learningFunction as lf
import common.calcFunctions as cf

#lf.unsupervised_learning()

unSupervised_data_path = 'data/csv/patients_Y6_full2.csv'

trainX_path = 'data/csv/patients_trainingX6.csv'
trainY_path = 'data/csv/patients_trainingY6.csv'
testX_path = 'data/csv/patients_testX6.csv'
testY_path = 'data/csv/patients_testY6.csv'

inference_testX_path = 'data/data.json'

lf.unsupervised_learning(unSupervised_data_path, dimension_reduction=2, clustering=2, n_component=2, n_cluster=5)
#lf.supervised_learning_training(trainX_path, trainY_path, testX_path, testY_path, datapps=2)
'''
predict_data = lf.supervised_learning_inference(inference_testX_path, isTraining=False)
print(predict_data)
predict_score = cf.setScore(predict_data)
print(predict_score)
'''
