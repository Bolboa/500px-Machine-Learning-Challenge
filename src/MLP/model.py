import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


'''
Define the layers of the Multi-Layer Perceptron.
'''
def layers(x, weights, biases):

	# hidden layer 1
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	
	# hidden layer 2
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	
	# output layer
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer


def use_model():

	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

	# network parameters
	learning_rate = 0.001
	training_epochs = 25
	batch_size = 100
	display_step = 1

	# 256 neurons for every hidden layer
	n_hidden_1 = 256
	n_hidden_2 = 256

	# 784 neurons for the input layer since each image is 28X28
	n_input = 784
	
	# 10 classications
	n_classes = 10

	X = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, n_classes])

	# initialize the weights of every layer
	weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
	}
 	
 	# initialize the bias for every layer
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	# define the model
	logits = layers(X, weights, biases)

	# define the cost function
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=logits,
			labels=y
		))

	# gradient descent optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# initializing the variables
	init = tf.global_variables_initializer()

	# define the saver for saving the model
	saver = tf.train.Saver()  

	# start tensforflow session
	with tf.Session() as sess:
		
		sess.run(init)

		# reload model
		saver.restore(sess, "/tmp/model3.ckpt")

		# extract the indices of the number 2
		two_idxs_list = np.where(mnist.test.labels[:, 2].astype(int) == 1)
		two_idxs = two_idxs_list[0][:10]

		# use the indices to extract the images of 2 and their corresponding label
		two_images = mnist.test.images[two_idxs]
		two_labels = mnist.test.labels[two_idxs]

		# create a fake label of 6 to assign to each number 2
		y_six = np.zeros((len(two_images), len(two_labels.T)))
		y_six[:, 6] = np.ones(len(two_images))

		# take the derivative of the loss function from the model
		# and evaluate it using the fake label
		im_derivative = tf.gradients(loss_op, X)[0]
		im_derivative = im_derivative.eval({X: two_images, 
                                    y: y_six })

		# we choose the epsilon values for which the score for the label 6
		# is above all the other scores
		epsilons = np.array([-0.15, -0.15, -0.15, -0.15, 
	                     -0.15, -0.15, -0.15, -0.15, 
	                     -0.15, -0.15]).reshape((10, 1))

		# calculate the noise and apply it to every image
		sign_values = np.sign(im_derivative)
		noise = epsilons * sign_values
		x_ad = two_images + noise

		# display the predictions for the original images and the adversarial images
		old_images = tf.argmax(logits, 1).eval({X:two_images})
		adversarial = tf.argmax(logits, 1).eval({X:x_ad})

		print("Labels of the original images: {}".format(old_images))
		print("Labels of the adversarial images: {}".format(adversarial))

		# initialize variable for column index
		jump = 3

		# initialize number of rows
		rows = len(two_images)

		pred = tf.nn.softmax(logits)

		# create a figure
		plt.figure(figsize=(17, 17))

		# plot orginal image -> sign of gradient -> adversarial image
		for i in range(rows):
			plt.subplot(rows, 3, jump-2).set_title("Confidence Label 6: " + str(sess.run(pred, feed_dict={X:two_images[i].reshape((1, 784))})[0][6] * 100) + "%")
			plt.subplots_adjust(top=2, bottom=0.01, hspace=0.5, wspace=0.4)
			plt.imshow(two_images[i].reshape(28,28), cmap='gray')

			plt.subplot(rows,3,jump - 1).set_title("Delta")
			plt.imshow(sign_values[i].reshape(28,28), cmap='gray')

			plt.subplot(rows, 3, jump).set_title("Confidence Label 6: " + str(sess.run(pred, feed_dict={X:x_ad[i].reshape((1, 784))})[0][6] * 100) + "%")
			plt.imshow(x_ad[i].reshape(28,28), cmap='gray')

			jump += 3

		plt.show()


use_model()