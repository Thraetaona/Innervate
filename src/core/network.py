# This file is part of Innervate, hereby provided under the MIT License.

import numpy as np


# # How to Instantiate: 
#     neural_network = Network([
#         Dense(28 * 28, 20, Sigmoid()),
#         Dense(20     , 10, Sigmoid()),
#     ])
# 
# # Training:
#     neural_network.train(...)
class Network(list):
    """ The main class abstracting the instantiation, training, and usage
    of a neural network supplied as a simple python list."""

    def __init__(self, network_list):
        super().__init__() # Initialize Python's built-in `list` class.

        # NOTE: `self` alone also refers to the "input" network (list);
        # e.g., in `[1, 2].add()`, add() will take `[1, 2]` in as a `self`.
        self._network = network_list


    def predict(self, input_matrix):
        prediction = input_matrix
        for layer in self._network: # Feed-Forward
            prediction = layer.forward(prediction)
        
        return prediction


    def backprop(self, first_gradient):
        gradient = first_gradient
        for layer in reversed(self._network):
            gradient = layer.backward(gradient, self._learning_rate)

        return gradient # Return the final (i.e., left-most) gradients


    def train(self, dataset, loss_type, epochs=1000, alpha=0.01, verbose=True):
        # Used internally by the backprop method.
        self._learning_rate = alpha
        x_train = dataset[0]; y_train = dataset[1]

        for epoch in range(epochs): # Iterate through training `epochs` times.

            error = 0
            for x, y in zip(x_train, y_train): # For each input-answer pair...
                """ The Forward-Propagation Pass """
                prediction = self.predict(x)

                # Calculate the average error (NOT needed to actually train):
                error += loss_type.loss(y, prediction)

                """ The Backward-Propagation Pass """
                gradient = self.backprop(loss_type.loss_prime(y, prediction))

            error /= len(x_train)
            if verbose:
                print(f"{epoch+1}/{epochs},\terror={error}")
        

    # TODO: Add support for an "integer" option where floats get translated
    # into integers.  Currently, this only converts floating-point to fixed-
    # point representations.
    #
    # NOTE: In fixed-point arithmetic, the "Q" notation is used to indicate
    # the minimum number of bits required to represent a range of values.
    # For example, signed Q0.7 uses 1 bit for the signdedness, 0 bit for 
    # the integer part, and 7 bits for the fractional part.  Similarly,
    # unsigned Q2.5 represents 2 integer and 5 fractional bits.
    # See: https://inst.eecs.berkeley.edu/~cs61c/sp06/handout/fixedpt.html
    def compress(self, integer_bits, fractional_bits=None):
        """ Translates the Network's trained parameters (i.e., weights
        and biases) to a fixed-point representation. """
        def _compress_params(params, integer_range, fractional_resolution):
            integer_min = integer_range[0]; integer_max = integer_range[1]

            # Calculate the current range of the params array
            params_min = np.min(params); params_max = np.max(params)

            # Noramlize (i.e., translate or scale)
            #compressed_params = np.interp(
            #    params,
            #    (params_min, params_max),
            #    (integer_min, integer_max)
            #)
            
            # Clip (discard overflows/underflows)
            compressed_params = np.clip(params, integer_min, integer_max)


            # Round the precision/resolution, if needed
            if fractional_resolution is not None:
                # Decimal-based precision (e.g., 1 for 0.1, 3 for 0.01, etc.)
                #compressed_params = np.round(
                #    compressed_params, decimals=fractional_resolution
                #)

                compressed_params = np.round(
                    compressed_params / fractional_resolution
                ) * fractional_resolution

                # By standard, -0.0 is equal to +0.0 anyhow
                compressed_params[compressed_params == -0.0] = 0.0

            return compressed_params

        # TODO: Foundation for integer-based scaling
        #w_min = np.min( [np.min(layer.weights) for layer in self._network] )
        #w_max = np.max( [np.max(layer.weights) for layer in self._network] )
        #b_min = np.min( [np.min(layer.biases) for layer in self._network] )
        #b_max = np.max( [np.max(layer.biases) for layer in self._network] )
        #global_min = min(w_min, b_min); global_max = max(w_max, b_max)
        
        resolution = 1 / (2 ** fractional_bits)
        integer_min = -(2 ** integer_bits)
        integer_max = (2 ** integer_bits) - 1

        for layer in self._network:
            # Compress the weights
            layer.weights = _compress_params(
                layer.weights, (integer_min, integer_max), resolution
            )

            # Compress the biases
            layer.biases = _compress_params(
                layer.biases, (integer_min, integer_max), resolution
            )


    # Basically, for each layer with `N` neurons connected to `M` _preceding_
    # neurons, we would have a 2-D array containing `N` 1-D arrays of M weights
    # (numbers) for each PRECEDING neuron's output (i.e., current input).
    # Similarly, each layer would also have a 1-column matrix of `N` biases.
    def save(self, folder_path, file_extension="txt"):
        FORMAT = str("%0.4f") # TODO: Don't hard-code the format.

        for i, layer in enumerate(self._network):
            params = layer()
            
            weights_file = f"{folder_path}/weights_{i+1}.{file_extension}"
            biases_file = f"{folder_path}/biases_{i+1}.{file_extension}"
            
            np.savetxt(weights_file, params["W"], delimiter=',', fmt=FORMAT)
            np.savetxt(biases_file, params["B"], delimiter=',', fmt=FORMAT)
