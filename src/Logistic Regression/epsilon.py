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

    # Graph Input.
    x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

    # Set model weights and bias.
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Construct model using the Softmax function.
    logits = tf.matmul(x, W) + b
    pred = tf.nn.softmax(logits)

    # Minimize error using cross entropy.
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    
    # Gradient Descent.
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    # Initializing all variables.    
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Reload model.
        saver.restore(sess, "/tmp/model.ckpt")

        # Initialize array that will store images of number 2.
        labels_of_2 = []
        
        # Get number 2 from mnist dataset.
        while mnist.test.next_batch(FLAGS.batch_size):

            # Get next batch.
            sample_image, sample_label = mnist.test.next_batch(FLAGS.batch_size)

            # Returns index of label.
            itemindex = np.where(sample_label == 1)

            # If image label is a number 2 store the image.
            if itemindex[1][0] == 2:
                labels_of_2.append(sample_image)
            else:
                continue

            # If there are 10 images stored then end the loop.
            if len(labels_of_2) == 10:
                break

        
        # Convert into a numpy array of shape [10, 784].
        adversarial = np.concatenate(labels_of_2, axis=0)

        # Generate 101 different epsilon values to test with.
        epsilon_res = 101
        eps = np.linspace(-1.0, 1.0, epsilon_res).reshape((epsilon_res, 1))

        # Labels for each image (used for graphing).
        labels = [str(i) for i in range(10)]

        # Set different colors for every Softmax hypothesis.
        num_colors = 10
        cmap = plt.get_cmap('hsv')
        colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
        
        # Create an empty array for our scores.
        scores = np.zeros((len(eps), 10))

        # Placeholder for target label.
        fake_label = tf.placeholder(tf.int32, shape=[10])

        # Setup the fake loss.
        fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=fake_label)

        # Calculate the gradient.
        gradients = tf.gradients(fake_loss, x)

        # Compute gradient value using the same Softmax model used to train the orginal model.
        # The gradient generates the values necessary to change the predictions of the number 2 to a number 6
        # with minimal cost.
        gradient_value = sess.run(gradients, feed_dict={x:adversarial, fake_label:np.array([6]*10)})

        for j in range(len(adversarial)):

            # Extract the sign of the gradient value for each image.
            sign = np.sign(gradient_value[0][j])

            # Apply every epsilon value along with the sign of the gradient to the image.
            for i in range(len(eps)):
                x_fool = adversarial[j].reshape((1, 784)) + eps[i] * sign

                # The scores are re-evaluated using the model and each 10 hypotheses are saved.
                scores[i, :] = logits.eval({x:x_fool})

            # Create a figure.
            plt.figure(figsize=(15, 15))
            plt.title("Image {}".format(j))

            # Loop through transpose of the scores to plot the effect of epsilon of every hypothesis.
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
