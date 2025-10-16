import numpy as np
import pytest
import torch

from src.dl.maths.transpose import numpy_transpose, torch_transpose

test_matrix = [([[1], [2], [3], [4]], [[1, 2, 3, 4]], (1, 4)),
               ([[5, 6, 7, 8]], [[5], [6], [7], [8]], (4, 1)),
               ([
                    [1, 2],
                    [3, 4]
                ],
                [
                    [1, 3],
                    [2, 4]
                ], (2, 2)),
               ([
                    [1, 2, 3],
                    [3, 4, 5],
                    [5, 6, 7],
                    [7, 8, 11]
                ],
                [
                    [1, 3, 5, 7],
                    [2, 4, 6, 8],
                    [3, 5, 7, 11]
                ], (3, 4))]


def get_test_matrix():
    return pytest.mark.parametrize("vector, exp_transpose, size", test_matrix)


@get_test_matrix()
def test_transpose_vector_numpy(vector, exp_transpose, size):
    print(f'Vector - \n{vector}')
    vector_t = numpy_transpose(vector)
    print(f'vector transpose - \n{vector_t}')
    assert np.array_equal(exp_transpose, vector_t), \
        f'Transpose - {vector_t} does not match with expected transpose - {exp_transpose} '
    assert np.array_equal(vector_t, np.array(vector).T), f'vector - {vector} Transpose Fails'
    assert vector_t.shape == size, f'size does not match'


@get_test_matrix()
def test_transpose_vector_tensor(vector, exp_transpose, size):
    print(f'Vector - \n{vector}')
    vector_t = torch_transpose(vector)
    print(f'vector transpose - \n{vector_t}')
    assert np.array_equal(exp_transpose, vector_t), \
        f'Transpose - {vector_t} does not match with expected transpose - {exp_transpose} '
    assert np.array_equal(vector_t, torch.tensor(vector).T), f'vector - {vector} Transpose Fails'
    assert vector_t.shape == size, f'size does not match'
