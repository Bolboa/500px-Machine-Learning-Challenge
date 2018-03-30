import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data



def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

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
A 2x2 pooling without any overlap.
'''
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1],
							padding='SAME')




def CNN(mnist):

	sess = tf.InteractiveSession()

	y = tf.placeholder(tf.float32, shape=[None, 10])
	x = tf.placeholder(tf.float32, shape=[None, 784])

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

	prob_drop = tf.placeholder(tf.float32)
	dropout = tf.nn.dropout(fully, prob_drop)

	# readout layer that will return the raw values
	# of our predictions
	W_readout = weight_variable([1024, 10])
	b_readout = bias_variable([10])

	y_conv = tf.matmul(dropout, W_readout) + b_readout

	# loss function
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))

	# train step using Adam Optimizer
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))

	accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver(tf.trainable_variables())

	for i in range(20001):
		batch = mnist.train.next_batch(100)
		if i%100:
			train_acc = accuracy.eval(feed_dict={
					x:batch[0], y:batch[1], prob_drop:1.0
				})
			print("step %d, training accuracy %g"%(i, train_acc))
		train_step.run(feed_dict={x:batch[0], y:batch[1], prob_drop:0.5})

	save_path = saver.save(sess, "/tmp/model2.ckpt")
	print("Model saved in file: %s" % save_path)


	print("test accuracy %g" % accuracy.eval({
			x: mnist.test.images,
			y: mnist.test.labels,
			prob_drop: 1.0
		}))


def main():
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	CNN(mnist)
main()
