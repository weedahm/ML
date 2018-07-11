import tensorflow as tf
import numpy as np
import csv

#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

class ThreeLayerNet:
    def __init__(self, train_X, train_Y, test_X, test_Y):
        self.trainX = train_X
        self.trainY = train_Y
        self.testX = test_X
        self.testY = test_Y
    
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
        X = tf.placeholder(tf.float32, [None, 33])
        Y = tf.placeholder(tf.float32, [None, 6])

        W1 = tf.Variable(tf.random_normal([33, 20], stddev=0.01))
        L1 = tf.nn.relu(tf.matmul(X, W1))

        W2 = tf.Variable(tf.random_normal([20, 10], stddev=0.01))
        L2 = tf.nn.relu(tf.matmul(L1, W2))

        W3 = tf.Variable(tf.random_normal([10, 6], stddev=0.01))

        model = tf.matmul(L2, W3)

        cost = tf.reduce_mean(tf.square(model - Y)) # Mean Square Error
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

        #########
        # Training
        #########
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        batch_size = 50
        total_batch = int(self.trainX.shape[0] / batch_size)

        for epoch in range(700):
            total_cost = 0

            for i in range(total_batch):
                batch_mask = np.random.choice(self.trainX.shape[0], batch_size)
                batch_xs = self.trainX[batch_mask]
                batch_ys = self.trainY[batch_mask]
                #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
                total_cost += cost_val

            print('Epoch:', '%04d' % (epoch + 1),
                'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

        print('최적화 완료!')

        #########
        # Testing
        #########
        print('MSE유사도: ', sess.run(cost, feed_dict={X: self.testX, Y: self.testY}))
        predict_model = sess.run(model, feed_dict={X: self.testX})
        predict_model = predict_model.round(3)

        f = open('output.csv', 'w', newline='')
        wr = csv.writer(f)
        for i in predict_model:
            wr.writerow(i)
        f.close()
        
        #predict_model.round(3)
        '''
        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도:', sess.run(accuracy,
                                feed_dict={X: self.testX,
                                        Y: self.testY}))
        '''
