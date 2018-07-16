import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

def inferenceNet(testX):
    input_size = int(testX.shape[1])
    hidden_size = int(input_size / 2)
    output_size = 6

    X = tf.placeholder(tf.float32, [None, input_size])
    #Y = tf.placeholder(tf.float32, [None, 6])

    W1 = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=0.01))
    L1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(X, W1)))

    W2 = tf.Variable(tf.random_normal([hidden_size, output_size], stddev=0.01))
    model = tf.matmul(L1, W2)

    #cost = tf.reduce_mean(tf.square(model - Y)) # Mean Square Error

    #optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    #########
    # Inference
    #########
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    save_path = "./patients_2layerNN/2layerNN.ckpt"
    saver.restore(sess, save_path)

    predict_model = sess.run(model, feed_dict={X: testX})
    predict_model = predict_model.round(1)

    #########
    # Writing
    #########
    f = open('inference_output.csv', 'w', newline='')
    wr = csv.writer(f)
    for i in predict_model:
        wr.writerow(i)
    f.close()
