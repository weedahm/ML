import os
import tensorflow as tf
import numpy as np
from .common import csv2data

script_path = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PATH = 'output/inference_output.csv'
MODEL_PATH = script_path + '/model/patients_2layerNN_two/2layerNN.ckpt'

def inferenceNet(testX):
    ######### Inference
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(MODEL_PATH+'.meta') # Load Network
        saver.restore(sess, MODEL_PATH) # Load Weight

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        #model = graph.get_operation_by_name("op_model") --- operation 으로 저장하는법?
        model = graph.get_tensor_by_name("op_model:0")

        predict_model = sess.run(model, feed_dict={X: testX})
        predict_model = predict_model.round(1)
    tf.reset_default_graph()

    ######### Writing
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    csv2data.set_data(OUTPUT_PATH, predict_model)

    predict_model[predict_model < 0] = 0
    predict_data = (predict_model * 2).round() / 2

    return predict_data
