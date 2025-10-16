import numpy as np
import torch


def numpy_transpose(vector):
    vector = np.array(vector)
    return np.transpose(vector)


def torch_transpose(vector):
    vector = torch.tensor(vector)
    return torch.transpose(vector, 0, 1)
