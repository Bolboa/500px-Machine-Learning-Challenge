import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd



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



def MLP(mnist):

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

		# training cycles
		for epoch in range(training_epochs):

			avg_cost = 0
			total_batch = int(mnist.train.num_examples/batch_size)

			# loop over all batches
			for i in range(total_batch):

				batch_xs, batch_ys = mnist.train.next_batch(batch_size)

				# minimize the cost
				_, c = sess.run([train_op, loss_op], feed_dict={X:batch_xs, y: batch_ys})

				# calculate the average cost
				avg_cost += c / total_batch

			# print the average cost at every epoch
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))


		# save model
		save_path = saver.save(sess, "/tmp/model3.ckpt")

		print("Optimization Finished")

		# apply the softmax classification to the model
		pred = tf.nn.softmax(logits)

		# list of booleans to determine the correct predictions
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

		# calculate the total accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print("Accuracy: ", accuracy.eval({X:mnist.test.images, y:mnist.test.labels}))

		# calculate the confusion matrix
		confusion = tf.confusion_matrix(labels=tf.argmax(mnist.test.labels, 1), predictions=tf.argmax(pred, 1))
		print(confusion.eval({X:mnist.test.images, y:mnist.test.labels}))

		# calculate the precision score
		precision = precision_score(
			tf.argmax(mnist.test.labels, 1).eval({
				X:mnist.test.images, y:mnist.test.labels
			}), 
			tf.argmax(pred, 1).eval({
				X:mnist.test.images, y:mnist.test.labels
			}), 
			average=None
		)

		print(precision)

		# calculate the recall score
		recall = recall_score(
			tf.argmax(mnist.test.labels, 1).eval({
				X:mnist.test.images, y:mnist.test.labels
			}),
			tf.argmax(pred, 1).eval({
				X:mnist.test.images, y:mnist.test.labels 
			}),
			average=None
		)

		print(recall)

		# save the predicted output along with the actual output
		df = pd.DataFrame()
		df["Predicted"] = tf.argmax(pred, 1).eval({X:mnist.test.images, y:mnist.test.labels})
		df["Actual"] = tf.argmax(mnist.test.labels, 1).eval({X:mnist.test.images, y:mnist.test.labels})
		df.to_csv("MLP.csv")



def main():
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
	MLP(mnist)
main()