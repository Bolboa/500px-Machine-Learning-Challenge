import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_integer('training_epochs', 20, 'Number of times training vectors are used once to update weights.')
flags.DEFINE_integer('batch_size', 1, 'Batch size. Must divide evenly into the data set sizes.')

def use_model():
    
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

    # set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

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
        adversarial = np.concatenate(labels_of_2, axis=0)

        # create a copy to graph the original image and compare its adversarial counterpart
        original_copy = np.concatenate(labels_of_2, axis=0)

        # these are the optimal epsilon values chosen through analyzing the graphs
        # generated by epsilon.py
        epsilons = np.array([-0.15, -0.15, -0.15, -0.15, 
                     -0.15, -0.15, -0.2, -0.2, 
                     -0.2, -0.2]).reshape((10, 1))
        
        # placeholder for target label
        fake_label = tf.placeholder(tf.int32, shape=[10])

        # setup the fake loss
        fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=fake_label)

        # calculate gradient
        gradients = tf.gradients(fake_loss, x)

        # compute gradient value using the same Softmax model used to train the orginal model,
        # the gradient generates the values necessary to change the predictions of the number 2 to a number 6
        # with minimal cost
        gradient_value = sess.run(gradients, feed_dict={x:adversarial, fake_label:np.array([6]*10)})

        # array to hold the sign of every gradient
        sign_values = []
        
        for j in range(len(adversarial)):

            # calculate the sign of the gradient
            sign = np.sign(gradient_value[0][j])

            # save all sign values
            sign_values.append(sign)

            noise = epsilons * sign

            # apply the noise to every image
            adversarial[j] = adversarial[j] + noise[j]

        # initialize variable for column index
        jump = 3

        # initialize number rows
        rows = len(original_copy)

        # plot orginal image -> sign of gradient -> adversarial image
        for i in range(len(original_copy)):
            plt.subplot(rows,3,jump - 2)
            plt.imshow(sess.run(x[0], feed_dict={x:original_copy}).reshape(28,28),cmap='gray')
            plt.subplot(rows,3,jump - 1)
            plt.imshow(sess.run(x[0], feed_dict={x:sign.reshape((1, 784))}).reshape(28,28),cmap='gray')
            plt.subplot(rows,3,jump)
            plt.imshow(sess.run(x[0], feed_dict={x:adversarial}).reshape(28,28),cmap='gray')
            jump = jump + 3
        
        plt.show()

        # after altering each image, have the model make a prediction on adversarial images
        classification_adversarial = sess.run(tf.argmax(pred, 1), feed_dict={x:adversarial})
        print(classification_adversarial)

        # after altering each image, have the model make a prediction on gradient sign
        classification_sign = sess.run(tf.argmax(pred, 1), feed_dict={x:sign_values})
        print(classification_sign)
        
use_model()

