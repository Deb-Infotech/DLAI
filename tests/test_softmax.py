import numpy
import pytest

from maths import softmax
from utilities import utilities

vectors_1D = [([1, 3, 5, 7]), ([-2, -1, 0, 1]), ([-10, -7, -4, -1])]
rand_vectors_1D = [(utilities.get_numpy_random_nd_array()), (utilities.get_numpy_random_nd_array(-20, 30, 70))]


def get_parametrization_random_1D():
    return pytest.mark.parametrize("vector1", rand_vectors_1D)


@pytest.mark.parametrize("vector1", vectors_1D)
def test_softmax_numpy(vector1):
    soft_max = softmax.get_numpy_softmax(vector1)
    sum_soft_max = numpy.sum(soft_max)
    assert round(sum_soft_max) == 1

    # x_dataset, y_dataset, x_label, y_label, graph_title, graph_file_name ##Tuple dataset for graph
    graph_info_tuple = (
        vector1,
        soft_max,
        f' X -> Vectors',
        f'Softmax $\sigma$',
        f'Softmax - $\sum \sigma$ = {round(sum_soft_max)}',
        f'softmax.png'
    )
    utilities.plot_save_graphs(graph_info_tuple)


@get_parametrization_random_1D()
def test_soft_max_numpy_random_1D(vector1):
    test_softmax_numpy(vector1)
