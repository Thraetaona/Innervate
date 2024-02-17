# Innervate: A simple and free-from-overcomplications implementation of
# artificial neural networks ("ANN") using only Python and NumPy.
#
# MIT License
# 
# Copyright (c) 2024 Fereydoun Memarzanjany
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from utils.data import *
from core.losses import *
from core.activations import *
from core.layers import *
from core.network import *

import numpy as np

EPOCHS = int(3)
LEARNING_RATE = float(0.01)


def main():
    x_train, y_train = preprocess_data(f"./datasets/mnist.npz")

    """ Network Instantiation """
    neural_network = Network([
        Dense(28 * 28, 20, Sigmoid()),
        Dense(20     , 10, Sigmoid()),
    ])

    """ Network Training """
    neural_network.train(
        (x_train, y_train),
        MSE(),
        epochs=EPOCHS,
        alpha=LEARNING_RATE
    )

    """ Sample Usage """
    output = neural_network.predict(x_train[0])

# Main harness function for driving the code.
if (__name__ == "__main__"): # The module is being run directly
    main()
else: # The module is being imported into another one
    pass