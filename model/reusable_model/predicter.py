import tensorflow as tf
import os

os.getcwdb

def getPrediction(image_data,export_path):
    print(export_path)
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], export_path)
        graph = tf.get_default_graph()
        return sess.run('label:0',
                        feed_dict={'imageToPredict:0': image_data})
