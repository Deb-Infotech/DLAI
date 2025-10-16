import numpy
import torch


def numpy_dot(vector1, vector2):
    vector1 = numpy.array(vector1)
    vector2 = numpy.array(vector2)
    return numpy.dot(vector1, vector2)  # numpy's dot performs matrix multiplication for 2D


def tensor_dot(vector1, vector2):
    vector1 = torch.tensor(vector1)
    vector2 = torch.tensor(vector2)
    return torch.matmul(vector1, vector2)
