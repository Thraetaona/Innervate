# This file is part of Innervate, hereby provided under the MIT License.

import numpy as np

INIT_WEIGHT_MIN = float(-0.5)
INIT_WEIGHT_MAX = float(+0.5)


class Layer:
    """ The generic class used to implement various types of layers
    (e.g., CNNs, RNNs, ANNs/Dense, etc.) as subclasses. """

    def forward():
        raise NotImplementedError(
            "Subclasses should implement a method for forward-propagation!"
        )

    def backward():
        raise NotImplementedError(
            "Subclasses should implement a method for backward-propagation!"
        )

class Dense(Layer):
    def __init__(self, input_size, output_size, activation_type):
        # NOTE: Uniform randomness yields much better training results.
        self.weights = np.random.uniform(
            INIT_WEIGHT_MIN, INIT_WEIGHT_MAX, (output_size, input_size)
        )
        # NOTE: We have to initialize with zeros; we want to avoid starting
        # with a biased, untrained network right from the beginning.
        self.biases = np.zeros((output_size, 1)) # Bias is a "column" vector.

        # Internal callbacks
        self.__activation = activation_type.forward
        self.__activation_prime = activation_type.backward

    def forward(self, input):
        self._input = input

        # In Mathematic Terms: a(W.X + B)
        return self.__activation(
            np.dot(self.weights, self._input) + self.biases
        )

    def backward(self, previous_output_gradient, learning_rate):
        # Calculate this layer's weights' gradients (i.e., the instantanous
        # rates of change calculated from the derivatves) to be used LOCALLY
        # to train this layer's weights.
        current_weights_gradient = np.dot(
            self.__activation_prime(previous_output_gradient),
            self._input.T # Transpose matrix of the original input
        )

        # Calculate this layer's output gradients (which will later become the
        # input to the PRECEDING layers' backpropagations) based on the
        # gradients of the layer following (i.e., to the
        # right-hand side) this one.
        #
        # This is NOT needed/used locally to train this layer's weights/biases.
        upcoming_input_gradient = np.dot(
            self.weights.T,
            previous_output_gradient
        )

        # "Train" the weights and biases based on how well (or poor) they had
        # performed in the previous run, according to our gradients.
        #
        # NOTE: We are using subtraction (-=) because gradients essentially
        # tell us how we can maximize our parameters as to make the prediction
        # perform WORSE; so, we do the "opposite" to optimize the prediction.
        self.weights -= learning_rate * current_weights_gradient
        self.biases -= learning_rate * previous_output_gradient

        # Return this layer's gradient vector so that it may be used to
        # calculate its preceding layer's gradient, similarly.
        return upcoming_input_gradient