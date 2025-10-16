import numpy as np


def get_numpy_softmax(vector1):
    vector1 = np.array(vector1)
    numerator = np.exp(vector1)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax
