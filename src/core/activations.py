# This file is part of Innervate, hereby provided under the MIT License.

import numpy as np


#     In basic terms, the "activation" function is a function which
# takes any vector of real number (-inf, +inf) and _clamps_ them
# as a [0, 1] real output; This clamping process is also referred
# to as "noramlization."
#
#     During the `forward` pass, inputs from one layer's neurons are fed to the
# to the subsequent layer's neurons; eventually, an intermediary output will
# be individually determined by each neuron based on their `input values`
# (X: x_1, x_2, x_3, ..., x_i) multiplied by their associated `input weights`
# (W: w_1, w_2, w_3, ..., w_i) with a final `neuron bias` (b) added afterward.
#     Finally, depending on where said intermediary output "falls" on the
# chosen activation function, our neuron in question will decide with what
# intensity between [0, 1] it should "fire" to the next layer's neurons.
#
#     Another important concept is "backpropagation."  In the backward pass,
# gradients (i.e., output derivatives of the error with respect to X, or dE/dX,
# given input derivatives of the error with respect to Y, or dE/dY)
#     Here, it is important to note that we are essentially going backwards and
# that the "input derivative" is actually being fed from the last layers all
# the way _back_ to the initial layers.  As for the very last layer, we will
# use the "Mean Square Error" (or MSE) derived formula (or any other loss
# function's derivative, like Binary Cross Entropy) as its input gradient.
#     One might be curious about the usefulness of backpropagation: backprop
# involves the actual training "magic" and is essentially a rate of change of
# the prediction error as a coefficient to update (i.e., adjust) the values
# of our network's weights and biases.
class Activation:
    """ The generic class used to implement various activation functions 
    (e.g., Sigmoid, Tanh, Step, ReLU, Softmax, etc.) as subclasses. """

    # NOTE: __init_subclass__ cannot be used here as it executes upon
    # each class' definition, not instantiation.
    def __init__(self, activation, activation_prime):
        """ Registers internal callback methods on class initialization. """
        
        self._activation = activation
        self._activation_prime = activation_prime

    def forward(self, input):
        self._input = input
        self._output = self._activation(self._input) # TODO: Use locally?
        return self._output

    # NOTE: Gradient = Derivative (i.e., the instantanous rate of change)
    # TODO: Could take in the alpha (learning rate) here and optimize further.
    def backward(self, previous_output_gradient):
        return np.multiply(
            previous_output_gradient, self._activation_prime(self._input)
        )
        


# Rectified Linear Unit ("ReLU") returns either a minimum
# of 0, or (otherwise) the >0 argument supplied.
#
# TODO: List advantages
class ReLU(Activation):
    def __init__(self):
        #           x + |x|
        # ReLU(x) = ------.
        #             2
        def __f(x):
            return np.maximum(0, x)

        # NOTE: ReLU's derivative is 1 for positive and 0 for negative inputs.
        # Technically, it is not differentiable at x=0 but, as a compromise, it
        # gets grouped with negative inputs to have a derivative of 0 at x=0:
        #     ReLU'(x) = { 1 | x >  0,
        #                { 0 | x <= 0.
        def __f_prime(x):
            # Using an index mask, set non-positive values' gradients to 0.
            results = x.copy(); results[__f(x) <= 0] = 0
            return results

        super().__init__(__f, __f_prime)


# TODO: List advantages
class Sigmoid(Activation):
    def __init__(self):
        #                   1  
        # Sigmoid(x) = ------------.
        #               1 + e^(-x)
        def __f(x):
            # Prevents overflow by conditionally using the appropriate
            # function format for negative values of x.
            return np.where(
                x >= 0, # The Condition
                1 / (1 + np.exp(-x)), # For non-negative values
                np.exp(x) / (1 + np.exp(x)) # For negative values
            )

        # Sigmoid'(x) = x * ( s(x) * (1 - s(x)) ).       
        def __f_prime(x):
            s = __f(x)
            return s * (1 - s)

        super().__init__(__f, __f_prime)


# The hyperbolic tangent ("Tanh") is a tangent (Tan) defined
# using a hyperbola rather than a circle.
#
# TODO: List advantages
class Tanh(Activation):
    def __init__(self):
        #                   1
        # Tanh(x) ~= --------------------
        #              1            1
        #            ----   +   ---------.
        #              x        3      1
        #                     ----  + ---
        #                       x   (...)
        def __f(x):
            return np.tanh(x)

        # Tanh'(x) = x * (1 - Tanh(x)^2).
        def __f_prime(x):
            return 1 - np.tanh(x)**2

        super().__init__(__f, __f_prime)