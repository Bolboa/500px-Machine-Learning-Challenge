import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('training_epochs', 25, 'Number of times training vectors are used once to update weights.')
flags.DEFINE_integer('batch_size', 1, 'Batch size. Must divide evenly into the data set sizes.')
flags.DEFINE_integer('display_step', 1, 'Tells function to print out progress after every epoch')


def use_model():
    
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # tf Graph Input
    x = tf.get_variable("input_image", shape=[1,784], dtype=tf.float32)
    y = tf.placeholder(shape=[1,10], name='input_label', dtype=tf.float32)  # 0-9 digits recognition => 10 classes

    # set model weights
    W = tf.get_variable("weights", shape=[784, 10], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b = tf.get_variable("biases", shape=[1, 10], dtype=tf.float32, initializer=tf.zeros_initializer())

    # construct model
    logits = tf.matmul(x, W) + b
    pred = tf.nn.softmax(logits)  # Softmax

    # minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    # initializing the variables    
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # reload model
        saver.restore(sess, "/tmp/model.ckpt")

        # get number 7 from mnist dataset
        sample_image, sample_label = mnist.test.next_batch(1)

        # assign the sample image to the variable
        sess.run(tf.assign(x, sample_image))
        # setup softmax
        sess.run(pred)

        # placeholder for target label
        fake_label = tf.placeholder(tf.int32, shape=[1])
        # minimize the cross entropy using the loss calculated using the target value 
        fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=fake_label)

        adversarial_step = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(fake_loss, var_list=[x])
        
        for i in range(100):
            sess.run(adversarial_step, feed_dict={fake_label:np.array([8])})
            sess.run(pred)
        plt.imshow(sess.run(x).reshape(28,28),cmap='gray')
        plt.show()
        
        x_in = np.expand_dims(x[0], axis=0)
        classification = sess.run(tf.argmax(pred, 1))
        print(classification)

use_model()

