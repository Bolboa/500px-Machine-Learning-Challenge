import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_integer('training_epochs', 30, 'Number of times training vectors are used once to update weights.')
flags.DEFINE_integer('batch_size', 1, 'Batch size. Must divide evenly into the data set sizes.')

def use_model():
    
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # tf Graph Input
    x = tf.get_variable("input_image", shape=[10,784], dtype=tf.float32)
    y = tf.placeholder(shape=[10,10], name='input_label', dtype=tf.float32)  # 0-9 digits recognition => 10 classes

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

        # initialize array that will store images of number 2
        labels_of_2 = []
        
        # get number 2 from mnist dataset
        while mnist.test.next_batch(FLAGS.batch_size):
            # get next batch
            sample_image, sample_label = mnist.test.next_batch(FLAGS.batch_size)
            # returns index of label
            itemindex = np.where(sample_label == 1)

            # if image label is a number 2 store the image
            if itemindex[1][0] == 2:
                labels_of_2.append(sample_image)
            else:
                continue

            # if there are 10 images stored then end the loop
            if len(labels_of_2) == 10:
                break

        # convert into a numpy array of shape [10, 784]
        labels_of_2 = np.concatenate(labels_of_2, axis=0)

        # assign the sample image to the variable
        sess.run(tf.assign(x, labels_of_2))
        # setup softmax
        sess.run(pred)

        # placeholder for target label
        fake_label = tf.placeholder(tf.int32, shape=[10])
        # setup the fake loss
        fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=fake_label)

        # minimize fake loss using gradient descent,
        # calculating the derivatives of the weight of the fake image will give the direction of weights necessary to change the prediction
        adversarial_step = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(fake_loss, var_list=[x])

        # continue calculating the derivative until the prediction changes for all 10 images
        for i in range(FLAGS.training_epochs):
            # fake label tells the training algorithm to use the weights calculated for number 6
            sess.run(adversarial_step, feed_dict={fake_label:np.array([6]*10)})
            sess.run(pred)

        plt.subplot(2,2,1)
        plt.imshow(sess.run(x[0]).reshape(28,28),cmap='gray')

        
        plt.show()

      
        x_in = np.expand_dims(x[0], axis=0)
        classification = sess.run(tf.argmax(pred, 1))
        print(classification)

use_model()

