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
    x = tf.placeholder(tf.float32, shape=[None, 784])
    #x = tf.get_variable("input_image", shape=[100,784], dtype=tf.float32)
    y = tf.placeholder(shape=[None,10], name='input_label', dtype=tf.float32)  # 0-9 digits recognition => 10 classes

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
            if len(labels_of_2) == 100:
                break

        
        # convert into a numpy array of shape [100, 784]
        labels_of_2 = np.concatenate(labels_of_2, axis=0)

        print(labels_of_2.shape)
        epsilon_res = 101
        eps = np.linspace(-1.0, 1.0, epsilon_res).reshape((epsilon_res, 1))
        labels = [str(i) for i in range(10)]

        num_colors = 10
        cmap = plt.get_cmap('hsv')
        colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]

        
        # Create an empty array for our scores
        scores = np.zeros((len(eps), 10))

        # placeholder for target label
        fake_label = tf.placeholder(tf.int32, shape=[100])
        # setup the fake loss
        fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=fake_label)

        gradients = tf.gradients(fake_loss, x)

        sess.run(pred, feed_dict={x:labels_of_2})
        
        gradient_value = sess.run(gradients, feed_dict={x:labels_of_2, fake_label:np.array([6]*100)})

        allScores = []
        sample_labels = labels_of_2


        sign = np.sign(gradient_value[0][0])
        for i in range(len(eps)):
            x_fool = labels_of_2[0].reshape((1, 784)) + eps[i] * sign
            scores[i, :] = logits.eval({x:x_fool})
        
        '''for j in range(len(labels_of_2)):
            sign = np.sign(gradient_value[0][j])
            
            for i in range(len(eps)):
                old_label = labels_of_2[j]
                new_label = labels_of_2[j].reshape((1, 784)) + eps[i] * sign
                sample_labels[...] = new_label
                
                scores[i,:] = logits.eval({x:sample_labels})[j]
                #sample_labels[...] = old_label
                
            allScores.append(scores)
        print(len(allScores[0].T))'''

        # Create a figure
        plt.figure(figsize=(10, 8))
        plt.title("Image {}".format(0))
        
        for k in range(len(scores.T)):
            plt.plot(eps, scores[:, k], 
                 color=colors[k], 
                 marker='.', 
                 label=labels[k])
            
        plt.legend(prop={'size':8})
        plt.xlabel('Epsilon')
        plt.ylabel('Class Score')
        plt.grid('on')
        plt.show()
        
      
        
use_model()

