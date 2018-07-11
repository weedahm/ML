import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
        X = tf.placeholder(tf.float32, [None, 40])
        Y = tf.placeholder(tf.float32, [None, 6])

        W1 = tf.Variable(tf.random_normal([40, 20], stddev=0.01))
        L1 = tf.nn.relu(tf.matmul(X, W1))

        W2 = tf.Variable(tf.random_normal([20, 6], stddev=0.01))

        model = tf.matmul(L1, W2)
        #L2 = tf.nn.relu(tf.matmul(L1, W2))

        #W3 = tf.Variable(tf.random_normal([12, 6], stddev=0.01))

        #model = tf.matmul(L2, W3)

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
        total_epoch = 300
        total_batch = int(self.trainX.shape[0] / batch_size)

        train_loss = []
        test_loss = []

        for epoch in range(total_epoch):
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

            temp_test_loss = sess.run(cost, feed_dict={X: self.testX, Y: self.testY})
            #print('test set MSE유사도: ', temp_test_loss)
            train_loss.append(total_cost / total_batch)
            test_loss.append(temp_test_loss)

        print('최적화 완료!')

        #########
        # Save Model
        #########
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./patients_2layerNN2/2layerNN.ckpt")

        import os
        print(os.getcwd())
        print("Model saved in file: ", save_path)

        #########
        # Testing
        #########
        print('MSE유사도: ', sess.run(cost, feed_dict={X: self.testX, Y: self.testY}))
        predict_model = sess.run(model, feed_dict={X: self.testX})
        predict_model = predict_model.round(1)

        #########
        # Writing
        #########
        f = open('output3.csv', 'w', newline='')
        wr = csv.writer(f)
        for i in predict_model:
            wr.writerow(i)
        f.close()

        #########
        # Draw plot
        #########
        x = np.arange(0, total_epoch, 1)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(x ,train_loss, label="train_loss")
        plt.plot(x ,test_loss, label="test_loss")
        plt.ylim(0, 8)
        plt.legend()
        plt.show()
