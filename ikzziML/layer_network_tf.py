import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from .common import csv2data

script_path = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PATH = 'output/output.csv'

class ThreeLayerNet:
    def __init__(self, train_X, train_Y, test_X, test_Y, size, model_path):
        self.trainX = train_X
        self.trainY = train_Y
        self.testX = test_X
        self.testY = test_Y
        self.input_size = size[0]
        self.output_size = size[1]
        self.model_path = model_path

    def Net(self):
        input_size = self.input_size
        hidden_size = 128
        hidden_size2 = 256
        output_size = self.output_size
        #hidden_size = int(input_size / 2)

        ######### Network Model
        X = tf.placeholder(tf.float32, [None, input_size], name="X")
        Y = tf.placeholder(tf.float32, [None, output_size])
        
        W1 = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=0.01))
        b1 = tf.Variable(tf.zeros([hidden_size]))
        L1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(X, W1) + b1))

        W2 = tf.Variable(tf.random_normal([hidden_size, hidden_size2], stddev=0.01))
        b2 = tf.Variable(tf.zeros([hidden_size2]))
        L2 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(L1, W2) + b2))

        W3 = tf.Variable(tf.random_normal([hidden_size2, hidden_size2], stddev=0.01))
        b3 = tf.Variable(tf.zeros([hidden_size2]))
        L3 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(L2, W3) + b3))

        W4 = tf.Variable(tf.random_normal([hidden_size2, hidden_size2], stddev=0.01))
        b4 = tf.Variable(tf.zeros([hidden_size2]))
        L4 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(L3, W4) + b4))

        W5 = tf.Variable(tf.random_normal([hidden_size2, output_size], stddev=0.01))
        b5 = tf.Variable(tf.zeros([output_size]))
        L5 = tf.matmul(L4, W5) + b5

        model = tf.identity(L5, name="op_model")

        cost = tf.reduce_mean(tf.square(model - Y)) # Mean Square Error
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
        #optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
        optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)

        ######### Training
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        batch_size = 30
        total_epoch = 200
        total_batch = int(self.trainX.shape[0] / batch_size)

        train_loss = []
        test_loss = []

        for epoch in range(total_epoch):
            total_cost = 0

            for i in range(total_batch):
                batch_mask = np.random.choice(self.trainX.shape[0], batch_size, replace=False)
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
        print("Model saved in file:", save_path)
        
        ######### Testing
        print('MSE유사도: ', sess.run(cost, feed_dict={X: self.testX, Y: self.testY}))
        predict_model = sess.run(model, feed_dict={X: self.testX})
        predict_model = predict_model.round(3)
        
        ######### Writing
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        csv2data.set_data(OUTPUT_PATH, predict_model)
        
        sess.close()

        ######### Draw plot
        x = np.arange(0, total_epoch, 1)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(x ,train_loss, label="train_loss")
        plt.plot(x ,test_loss, label="test_loss")
        plt.ylim(0, 5)
        plt.legend()
        plt.show()
