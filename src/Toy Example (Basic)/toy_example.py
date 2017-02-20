'''
A toy example for understanding the process behind breaking linear classifiers.
'''

import numpy
import math


# sigmoid function
def sigmoid(z):
    return 1/(1 + math.exp(-z))


# classifier probability
def probability(sig_val):
    if sig_val < 0.5:
        result = "Classifier is {}% certain that value is class 0".format(round((1-sig_val)*100, 2))
        print(result)
    else:
        result = "Classifier is {}% certain that value is class 1".format(round(sig_val*100, 2))
        print(result)


def main():
    # training set
    x = numpy.array([2, -1, 3, -2, 2, 2, 1, -4, 5, 1])
    # weights
    w = numpy.array([-1, -1, 1, -1, 1, -1, 1, 1, -1, 1])

    # dot product
    prod = numpy.dot(x, w)
    # get the classifier probability
    probability(sigmoid(prod))

    # increase training data by a fraction of each weight
    xad = x + 0.5*w

    # get the new classifier probability
    prod = numpy.dot(xad, w)
    probability(sigmoid(prod))

main()