import learningFunction as lf

#lf.unsupervised_learning()

trainX_path = 'data/csv/patients_trainingX5.csv'
trainY_path = 'data/csv/patients_trainingY5.csv'
testX_path = 'data/csv/patients_testX5.csv'
testY_path = 'data/csv/patients_testY5.csv'
#lf.supervised_learning_training(trainX_path, trainY_path, testX_path, testY_path, datapps=2)
lf.supervised_learning_inference(testX_path)
