# This file is part of Innervate, hereby provided under the MIT License.

import numpy as np


# Average error made during a network's prediction (i.e., how "incorrect" the
# prediction was) is calculated by one of the |loss| function implementations.
#
# However, more important than the loss itself is the loss' derivative; it
# tells us how much our incorrection has "changed" since the last mistake
# happend.  In other words, we can use that as an estimate or coefficient
# to adjust our neural network's parameters (weights and biases).
#
# NOTE: y_hat (y^) is the predicted value, while y is the "actual" answer.
class Loss: # _
    """ The generic class used to implement various loss subclasses
    (e.g., Mean Square Error, Binary Cross Entropy, etc.). """

    # NOTE: __init_subclass__ cannot be used here as it executes upon
    # each class' definition, not instantiation.
    def __init__(self, loss, loss_prime):
        """ Registers internal callback methods on class initialization. """
        
        self._loss = loss
        self._loss_prime = loss_prime

    def loss(self, y, y_hat):
        return self._loss(y, y_hat)

    def loss_prime(self,  y, y_hat):
        return self._loss_prime(y, y_hat)



# Mean Square Error
#
# TODO: List advantages
class MSE(Loss):
    def __init__(self):
        # MSE(x) = mean( (y - y_hat)^2 )
        def __f(y, y_hat):
            return np.mean(
                np.power(y - y_hat, 2)
            )

        # MSE'(x) = -2 * (y - y_hat) / y[size]
        def __f_prime(y, y_hat):
            return -2 * (y - y_hat)/np.size(y)

        super().__init__(__f, __f_prime)


# Binary Cross Entropy
#
# TODO: List advantages
class BCE(Loss):
    def __init__(self):
        # BCE(x) = TODO
        def __f(y, y_hat):
            return np.mean(
                -y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat)
            )

        # BCE'(x) = TODO
        def __f_prime(y, y_hat):
            return ( (1 - y)/(1 - y_hat) - y/y_hat ) / np.size(y)

        super().__init__(__f, __f_prime)