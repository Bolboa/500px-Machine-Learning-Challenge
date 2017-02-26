import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def use_model():

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("/tmp/model.ckpt")
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    all_vars = tf.get_collection('vars')
    for v in all_vars:
        v_ = sess.run(v)
        print(v_)



def main():
    use_model()

main()