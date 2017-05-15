# Adversarial Images - Part 1

This project is an experiment on breaking linear classifiers. In this particular example, the linear classifier implemented is the logistic regression function.

## Description

The method used to generate adversarial images was the *gradient sign method*.

![Alt text](/img/gsm.PNG)

The procedure is straightforward. First, the *mnist* data set is loaded and a logistic regression function using the **Softmax** model is trained to classify each image.
