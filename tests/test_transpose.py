import numpy
import numpy as np
import pytest

from maths.transpose import numpy_transpose


@pytest.mark.parametrize("vector, exp_transpose",
                         [([[1], [2], [3], [4]], [[1, 2, 3, 4]]), ([[5, 6, 7, 8]], [[5], [6], [7], [8]]),
                          ([
                               [1, 2],
                               [3, 4]
                           ],
                           [
                               [1, 3],
                               [2, 4]
                           ]),
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
                           ])])
def test_numpy_transpose_vector(vector, exp_transpose):
    vector = numpy.array(vector)
    print(f'Vector - \n{vector}')
    vector_t = numpy_transpose(vector)
    print(f'vector transpose - \n{vector_t}')
    assert np.array_equal(exp_transpose, vector_t), \
        f'Transpose - {vector_t} does not match with expected transpose - {exp_transpose} '
    assert np.array_equal(vector_t, vector.T), f'vector - {vector} Transpose Fails'
