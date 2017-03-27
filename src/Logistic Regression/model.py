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
    #x_placeholder = tf.placeholder(tf.float32, shape=[1, 784])
    #assign_x_op = x.assign(x_placeholder).op
    
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
        saver.restore(sess, "/tmp/model.ckpt")
        
        sample_image, sample_label = mnist.test.next_batch(1)
        #print(sess.run(x))
        #print(x.eval())
        sess.run(tf.assign(x, sample_image))
        sess.run(pred)
    
        #print(sess.run(x))
        #print(x.eval())
        
        fake_label = tf.placeholder(tf.int32, shape=[1])
        fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=fake_label)

        adversarial_step = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(fake_loss, var_list=[x])

        sess.run(adversarial_step, feed_dict={fake_label:np.array([8])})
        plt.imshow(sess.run(x).reshape(28,28),cmap='gray')
        sess.run(pred)
        
        #x_in = np.expand_dims(mnist.test.images[0], axis=0)
        #y_in = np.expand_dims(mnist.test.labels[0], axis=0)
        #classification = sess.run(tf.argmax(pred, 1), feed_dict={x:x_in})
        #print(classification)

        #accuracy = tf.reduce_mean(tf.cast(classification, tf.float32))
        #print("Accuracy:", accuracy.eval({x: x_in , y: y_in}))



use_model()

