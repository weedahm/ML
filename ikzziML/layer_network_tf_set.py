import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from .common import csv2data

script_path = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PATH = 'output/output_set.csv'

class TwoLayerNet:
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
        output_size = self.output_size
        hidden_size = 64

        ######### Network Model
        X = tf.placeholder(tf.float32, [None, input_size], name="X")
        Y = tf.placeholder(tf.float32, [None, output_size])

        W1 = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=0.01))
        b1 = tf.Variable(tf.zeros([hidden_size]))
        L1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(X, W1) + b1))
        
        W2 = tf.Variable(tf.random_normal([hidden_size, output_size], stddev=0.01))
        b2 = tf.Variable(tf.zeros([output_size]))
        L2 = tf.matmul(L1, W2) + b2

        model = tf.identity(L2, name="op_model") 

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))

        optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)

        ######### Training
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        batch_size = 50
        total_epoch = 50
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

        print('최적화 완료!')

        ######### Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, self.model_path)
        print("Model saved in file: ", save_path)

        ######### Testing
        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('test 정확도:', sess.run(accuracy, feed_dict={X: self.testX, Y: self.testY})*100, '%')
        print('training 정확도:', sess.run(accuracy, feed_dict={X: self.trainX, Y: self.trainY})*100, '%')
        
        softmax_model = tf.nn.softmax(model)
        predict_model = sess.run(softmax_model, feed_dict={X: self.testX})
        predict_model = predict_model.round(2)
        #predict_model = np.argmax(predict_model, axis=1)
        
        sess.close()
        
        ######### Writing
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        csv2data.set_data(OUTPUT_PATH, predict_model)

        ######### Draw plot
        x = np.arange(0, total_epoch, 1)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(x ,train_loss, label="train_loss")
        plt.plot(x ,test_loss, label="test_loss")
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

        
