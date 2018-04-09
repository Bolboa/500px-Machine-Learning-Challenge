import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data



def layers(x, weights, biases):
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

	return out_layer


def use_model():

	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


	learning_rate = 0.001
	training_epochs = 25
	batch_size = 100
	display_step = 1

	n_hidden_1 = 256
	n_hidden_2 = 256
	n_input = 784
	n_classes = 10

	X = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, n_classes])

	weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
	}
 
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	logits = layers(X, weights, biases)

	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=logits,
			labels=y
		))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# initializing the variables
	init = tf.global_variables_initializer()

	saver = tf.train.Saver()  


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

		# create labels for each type of image
		labels = [str(i) for i in range(10)]

		# define the graph colors
		num_colors = 10
		cmap = plt.get_cmap('hsv')
		colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]

		# generate 101 epsilon values from -1.0 to 1.0
		epsilon_res = 101
		eps = np.linspace(-1.0, 1.0, epsilon_res).reshape((epsilon_res, 1))

		# empty array to hold our scores
		scores = np.zeros((len(eps), 10))

		# loop through each image of 2
		for j in range(len(two_images)):

			# reshape the image
			img = two_images[j].reshape((1, 784))

			# extract the sign of the derivative of the loss function
			sign = np.sign(im_derivative[j])

			# apply every epsilon value to each image
			for i in range(len(eps)):

				# equation for adding noise
				fool = img + eps[i] * sign
				scores[i, :] = logits.eval({X:fool})
		
			# define figure
			plt.figure(figsize=(10, 8))
			plt.title("Image {}".format(j))

			# plot the score for every epsilon value
			for k in range(len(scores.T)):
				plt.plot(eps, scores[:, k],
						color=colors[k],
						marker='.',
						label=labels[k]
						)

			# display graph
			plt.legend(prop={'size':8})
			plt.xlabel('Epsilon')
			plt.ylabel('Class Score')
			plt.grid('on')
			plt.show()



use_model()



