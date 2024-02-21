# This file is part of Innervate, hereby provided under the MIT License.

import numpy as np

INIT_WEIGHT_MIN = float(-0.5)
INIT_WEIGHT_MAX = float(+0.5)


class Layer:
    """ The generic class used to implement various types of layers
    (e.g., CNNs, RNNs, ANNs/Dense, etc.) as subclasses. """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        """
        if not hasattr(cls, "weights"):
            raise TypeError(f"{cls.__name__} must have a 'weights' attribute!")
        elif not hasattr(cls, "biases"):
            raise TypeError(f"{cls.__name__} must have a 'biases' attribute!")
        """

    def forward():
        """ Feeds the given input through the Layer and returns an output. """

        raise NotImplementedError(
            "Subclasses should implement a method for forward-propagation!"
        )

    def backward():
        """ Trains the Layer's parameters using the given output gradient. """
        
        raise NotImplementedError(
            "Subclasses should implement a method for backward-propagation!"
        )

    def __len__():
        """ Returns the number of neurons in the Layer. """

        raise NotImplementedError(
            "Subclasses should implement the len() method for "
            "returning the Layer's number of neurons!"
        )

    # NOTE: I originally wanted to use __dir__ / dir() here, but python3
    # always forces you to return lists (not even dictionaries) from it.
    def __call__():
        """ Returns the Layer's parameters ([W]eights & [B]iases matrices). """

        raise NotImplementedError(
            "Subclasses should be callable (using __call__) "
            "as to return the Layer's parameters."
        )


class Dense(Layer):
    def __init__(self, input_size, output_size, activation_type):
        # NOTE: Uniform randomness yields much better training results.
        self.weights = np.random.uniform(
            -1, +1, (output_size, input_size)
        )
        # NOTE: We have to initialize with zeros; we want to avoid starting
        # with a biased, untrained network right from the beginning.
        self.biases = np.zeros((output_size, 1)) # Bias is a "column" vector.

        # Internal callbacks
        self._activation = activation_type.forward
        self._activation_prime = activation_type.backward

    def forward(self, input):
        self._input = input

        # In Mathematic Terms: a(W.X + B)
        return self._activation(
            np.dot(self.weights, self._input) + self.biases
        )

    def backward(self, previous_output_gradient, learning_rate):
        # Calculate this layer's weights' gradients (i.e., the instantanous
        # rates of change calculated from the derivatves) to be used LOCALLY
        # to train this layer's weights.
        current_weights_gradient = np.dot(
            self._activation_prime(previous_output_gradient),
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
    
    def __len__(self):
        return self.weights.shape[0] # 0 = Rows.
    
    def __call__(self):
        return { # Returns a dictionary of the...
            "W": self.weights.tolist(), # Weights Matrix; and
            "B": self.biases.tolist() # Biases Matrix.
        }