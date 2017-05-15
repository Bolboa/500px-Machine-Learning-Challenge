# Adversarial Images - Part 1

This project is an experiment on breaking linear classifiers. In this particular example, the linear classifier implemented is the logistic regression function.

## Description

The method used to generate adversarial images was the *gradient sign method*.

![Alt text](/img/gsm.png)

The procedure is straightforward. First, the *mnist* data set is loaded and a logistic regression function using the **Softmax** model is trained to classify each image.

Once the model is trained, the **Softmax** function will contain the necessary weight distribution to make a prediction on any image it is given. The crucial step now is to calculate the gradient of each image. 

Each image is a number 2 and the goal of this project is to make the model think an image of a number 2 is in fact a number 6. In order to do this, the gradient of each image must be calculated using a fake label. The gradient function is using the weight distribution of the already trained **Softmax** function except now it is being fed an image of a 2 with a label of 6. It will try to minimize the cost of these images by edging the weight distribution of these images closer to a number 6. In doing so, the gradient function will return the new weight distribution of these fake images, and all that is needed is to take the sign of this weight distribution and change every image pixel in the original image in accordance to the sign of the weight distribution.

Below is an example of this process:

![Alt text](/img/panda.png)
