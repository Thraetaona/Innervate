# This file is part of Innervate, hereby provided under the MIT License.

import numpy as np

COLOR_DEPTH = int(4)
PIXEL_RANGE = int( (2**COLOR_DEPTH) - 1 )
NUM_LABELS = int(10)

def preprocess_data(dataset_file, limit=None):

    with np.load(dataset_file) as dataset:
        (images, labels) = dataset["data"], dataset["target"]

        # Resize and flatten all the 2-D images into 1-D arrays
        # with the same width X height.
        images = np.reshape(
            images, 
            (images.shape[0], 8*8, 1)
        )

        #digits.images.reshape((len(digits.images), -1))
        #images = images.reshape((images.shape[0], -1))

        # Clamp all the images' pixel values on a grayscale [0, 1] range
        # by recasting them as floats and dividing by their color depth.
        images = images.astype("float32") / PIXEL_RANGE

        # Encode the labels as one-hot encoded representations by indexing
        # the (I)dentity matrix using `labels`.
        # E.g., label of the number 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
        labels = np.eye(NUM_LABELS)[labels]
        labels = labels.reshape(labels.shape[0], NUM_LABELS, 1)

        return (images[:limit], labels[:limit])