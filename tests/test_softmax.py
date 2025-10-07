import numpy
import pytest

import maths.softmax

vectors_1D = [([1, 3, 5, 7]), ([-1, 4, -2, 9]), ([5, -2, 4, -2])]


@pytest.mark.parametrize("vector1", vectors_1D)
def test_softmax_numpy(vector1):
    softmax = maths.softmax.get_numpy_softmax(vector1)
    assert round(numpy.sum(softmax)) == 1
