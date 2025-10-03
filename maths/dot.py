import numpy
import torch


def numpy_dot(vector1, vector2):
    vector1 = numpy.array(vector1)
    vector2 = numpy.array(vector2)
    return numpy.dot(vector1, vector2)


def tensor_dot(vector1, vector2):
    vector1 = torch.tensor(vector1)
    vector2 = torch.tensor(vector2)
    return vector1 @ vector2
