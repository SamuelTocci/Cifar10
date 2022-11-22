import ssl
import numpy as np

import tensorflow as tf

from scipy import optimize

import utils

LABELS = 10

ssl._create_default_https_context = ssl._create_unverified_context

# Load in the data
cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = np.reshape(x_train, (50000, 1, 3072))

print("Data shape: ", x_train.shape)

print(x_train)


### CODE ASSIGNMENT STARTS HERE ###

def costFunction(theta, x, y, lambda_):
    # Initialize some useful values
    m = y.size

    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)

    J = 0
    grad = np.zeros(theta.shape)

    h = utils.sigmoid(np.dot(x, theta))
    lambdaCost = (lambda_ / (2 * m)) * sum(theta[1:] ** 2)
    lambdaGradient = (lambda_ / m) * theta[1:]
    J = (1 / m) * (np.dot(-y, np.log(h)) - np.dot(1 - y, np.log(1 - h))) + lambdaCost
    grad[0] = (1 / m) * np.dot(h - y, x[:, 0])
    grad[1:] = (1 / m) * np.dot(h - y, x[:, 1:]) + lambdaGradient

    return J, grad


def oneVsAll(x, y, num_labels, lambda_):
    # Some useful variables
    m, n = x.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    x = np.concatenate([np.ones((m, 1)), x], axis=1)

    # ====================== YOUR CODE HERE ======================
    for c in range(num_labels):
        # setting the initial theta
        theta_initial = np.zeros(n + 1)

        # Set options for minimize. This indicates the max number of iterations to perform.
        options = {'maxiter': 50}

        # Run minimize to obtain the optimal theta. This function will return a class object where theta is in `res.x` and cost in `res.fun`
        res = optimize.minimize(costFunction,  # function to be minimized
                                theta_initial,  # initial guess
                                (x, (y == c), lambda_),  # extra arguments
                                jac=True,
                                # Method for computing the gradient vector. It is set to true so it returns a tuple containing objective function and gradient
                                method='TNC',  # truncated newton algorithm
                                options=options)
        all_theta[c] = res.x

    # ============================================================
    return all_theta


### CODE ASSIGNMENT STOPS HERE ###

lambda_ = 0.1
all_theta = oneVsAll(x_train, y_train, LABELS, lambda_)
