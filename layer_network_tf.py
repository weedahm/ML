import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

class ThreeLayerNet:
    def __init__(self, train_X, train_Y, test_X, test_Y, size, save_model_path):
        self.model_path = save_model_path
        self.output_path = 'output/output.csv'
        self.trainX = train_X
        self.trainY = train_Y
        self.testX = test_X
        self.testY = test_Y
        self.input_size = size[0]
        self.output_size = size[1]
    
    '''
    def dataSetting(self, dataRatio = 0.8):
        #batch_mask = np.random.choice(self.X.shape[0], self.X.shape[0] * dataRatio)
        self.trainX = self.X[:self.X.shape[0] * dataRatio]
        self.trainY = self.X[:self.X.shape[0] * dataRatio]
        self.testX = self.X[:self.X.shape[0] * dataRatio]
        self.testY = self.X[:self.X.shape[0] * dataRatio]
        tf.data.Dataset.from_tensor_slices(self.X)
    '''

    def Net(self):
        input_size = self.input_size
        output_size = self.output_size
        hidden_size = int(input_size / 2)
        #hidden_size = 256
        #hidden2_size = int(hidden_size / 2)

        X = tf.placeholder(tf.float32, [None, input_size])
        Y = tf.placeholder(tf.float32, [None, output_size])

        W1 = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=0.01))
        L1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(X, W1)))

        W2 = tf.Variable(tf.random_normal([hidden_size, output_size], stddev=0.01))
        model = tf.matmul(L1, W2)
        #L2 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(L1, W2)))

        #W3 = tf.Variable(tf.random_normal([hidden2_size, output_size], stddev=0.01))

        #model = tf.matmul(L2, W3)

        cost = tf.reduce_mean(tf.square(model - Y)) # Mean Square Error
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
        #optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
        optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)

        ######### Training
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        batch_size = 30
        total_epoch = 150
        total_batch = int(self.trainX.shape[0] / batch_size)

        train_loss = []
        test_loss = []

        for epoch in range(total_epoch):
            total_cost = 0

            for i in range(total_batch):
                batch_mask = np.random.choice(self.trainX.shape[0], batch_size)
                batch_xs = self.trainX[batch_mask]
                batch_ys = self.trainY[batch_mask]
                _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
                total_cost += cost_val

            temp_test_loss = sess.run(cost, feed_dict={X: self.testX, Y: self.testY})
            train_loss.append(total_cost / total_batch)
            test_loss.append(temp_test_loss)
            
            print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', total_cost / total_batch, ', test cost = ', temp_test_loss)
            #    'Avg. cost =', '{:.3f}'.format(total_cost / total_batch), ', test cost = ', '{:.3f}'.format(temp_test_loss))

        print('최적화 완료!')

        ######### Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, self.model_path)

        import os
        print(os.getcwd())
        print("Model saved in file: ", save_path)

        ######### Testing
        print('MSE유사도: ', sess.run(cost, feed_dict={X: self.testX, Y: self.testY}))
        predict_model = sess.run(model, feed_dict={X: self.testX})
        predict_model = predict_model.round(3)

        ######### Writing
        filename = self.output_path
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = open(filename, 'w', newline='')
        wr = csv.writer(f)
        for i in predict_model:
            wr.writerow(i)
        f.close()

        ######### Draw plot
        x = np.arange(0, total_epoch, 1)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(x ,train_loss, label="train_loss")
        plt.plot(x ,test_loss, label="test_loss")
        plt.ylim(0, 50)
        plt.legend()
        plt.show()
