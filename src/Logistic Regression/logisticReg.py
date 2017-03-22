from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('training_epochs', 25, 'Number of times training vectors are used once to update weights.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the data set sizes.')
flags.DEFINE_integer('display_step', 1, 'Tells function to print out progress after every epoch')

def logistic_regression():
    
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

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
    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    # launch the graph
    with tf.Session() as sess:
        #mnist.test.images[0] += 0.5*weight

        sess.run(init)

        # extract first image and its corresponding label
        first_image = mnist.test.images[0] * 255
        first_label = mnist.test.labels[0]

        # reshape into a 28x28 image
        pixels = first_image.reshape((28, 28))

        # training cycle
        for epoch in range(FLAGS.training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples/FLAGS.batch_size)
            # loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})

                # compute average loss
                avg_cost += c / total_batch
            # display logs per epoch step
            if (epoch + 1) % FLAGS.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)
        print("Optimization Finished!")

        # list of booleans to determine the correct predictions
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        print(correct_prediction.eval({x:mnist.test.images, y:mnist.test.labels}))

        # calculate total accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        # display predicted label
        learned_label = sess.run(y, {x: mnist.test.images, y: mnist.test.labels})[0]
        learned_image = sess.run(x, {x: mnist.test.images, y: mnist.test.labels})[0]
        learned_image = learned_image.reshape((28,28))
        og_weights = sess.run(W[:,0])

        # title of image will be the predicted label
        predicted_label = learned_label.tolist().index(1)
        plt.suptitle(predicted_label)
        plt.imshow(learned_image, cmap='gray')
        plt.show()



def main(_):
    logistic_regression()
if __name__ == '__main__':
  tf.app.run()

