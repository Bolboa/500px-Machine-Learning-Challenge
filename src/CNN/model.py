import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


'''
A weight function for defining the size of the filters.
'''
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


'''
A bias function for defining the number of biases needed for each layer.
'''
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


'''
Convolution is done with a stride of 1 (overlap) and with zero-padding
in order to keep the output matrix the same dimensions as the input matrix.
'''
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


'''
A 2x2 pooling filter without any overlap.
'''
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1],
							padding='SAME')



def use_model():

	# load dataset
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

	# interactive session
	sess = tf.InteractiveSession()

	# data and labels placeholder
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y = tf.placeholder(tf.float32, shape=[None, 10])

	# 32 filters of size 5x5 and 32 biases,
	# the filters are used to create 32 feature maps
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	x_img = tf.reshape(x, [-1, 28, 28, 1])

	# first layer activated using a Relu activation function
	conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
	pool1 = max_pool_2x2(conv1)

	# 64 filters of size 5x5
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	# second layer
	conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
	pool2 = max_pool_2x2(conv2)

	# fully connected layer with 1024 neurons
	W_fully = weight_variable([7 * 7 * 64, 1024])
	b_fully = bias_variable([1024])

	pool2flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
	fully = tf.nn.relu(tf.matmul(pool2flat, W_fully) + b_fully)

	# dropout layer removes dead neurons
	prob_drop = tf.placeholder(tf.float32)
	dropout = tf.nn.dropout(fully, prob_drop)

	# readout layer that will return the raw values
	# of our predictions
	W_readout = weight_variable([1024, 10])
	b_readout = bias_variable([10])

	y_conv = tf.matmul(dropout, W_readout) + b_readout

	# loss function
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))

	# restore the trained CNN model
	saver = tf.train.Saver()
	saver.restore(sess, "/tmp/model2.ckpt")

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
	im_derivative = tf.gradients(cross_entropy, x)[0]
	im_derivative = im_derivative.eval({x: two_images, 
                                    y: y_six, 
                                    prob_drop: 1.0})

	# we choose the epsilon values for which the score for the label 6
	# is above all the other scores
	epsilons = np.array([-0.2, -0.5, -0.25, -0.25, 
                     -0.3, -0.49, -0.3, -0.7, 
                     -0.5, -0.5]).reshape((10, 1))

	# calculate the noise and apply it to every image
	noise = epsilons * np.sign(im_derivative)
	x_ad = two_images + noise

	# display the predictions for the original images and the adversarial images
	old_images = tf.argmax(y_conv, 1).eval({x:two_images, prob_drop:1.0})
	adversarial = tf.argmax(y_conv, 1).eval({x:x_ad, prob_drop:1.0})

	print("Labels of the original images: {}".format(old_images))
	print("Labels of the adversarial images: {}".format(adversarial))

	# initialize variable for column index
	jump = 3

	# initialize number of rows
	rows = len(two_images)

	# create a figure
	plt.figure(figsize=(17, 17))

	# plot orginal image -> sign of gradient -> adversarial image
	for i in range(rows):
		x_float = tf.to_float(x) / 255
		x_img = tf.reshape(x_float, [-1, 28, 28, 1])
		print(x_img)
		#prediction = tf.argmax(y_conv, 1)
		#print(sess.run(prediction, feed_dict={x:x_img}))
		#output = sess.run([prediction], feed_dict={x:two_images[i].reshape((1, 784))})
		#print(output)




use_model()

