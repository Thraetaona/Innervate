# This file is part of Innervate, hereby provided under the MIT License.


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