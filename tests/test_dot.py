import numpy
import pytest
import torch

from maths import dot

vectors_1D = [([1, 3, 5, 7], [2, 4, 6, 8], (4,), 100), ([-1, 4, -2, 9], [5, -2, 4, -2], (4,), -39)]
vectors_2D = [([[1, 3, 5, 7]], (1, 4), [[2], [4], [6], [8]], (4, 1), (1, 1), [[100]]),
              ([[-1, 4, -2, 9]], (1, 4), [[5], [-2], [4], [-2]], (4, 1), (1, 1), [[-39]]),
              ([[-1, 4, -2, 9], [-1, 4, -2, 9]], (2, 4), [[5], [-2], [4], [-2]], (4, 1), (2, 1), [[-39], [-39]])
              ]


def get_test_vectors_1D():
    return pytest.mark.parametrize("vector1, vector2, shape, expected_value", vectors_1D)


def get_test_vectors_2D():
    return pytest.mark.parametrize("vector1,shape1,  vector2, shape2, dot_exp_shape, expected_value", vectors_2D)


@get_test_vectors_1D()
def test_dot_numpy_1D(vector1, vector2, shape: tuple, expected_value: int):
    print(f'vector1 - \n {vector1}')
    print(f'vector2 - \n {vector2}')
    vector_dot = dot.numpy_dot(vector1, vector2)
    print(f'vector dot - \n{vector_dot}')
    assert type(vector_dot) == numpy.int64
    assert isinstance(vector_dot, numpy.int64)
    assert vector_dot.shape == ()
    shape_v1 = numpy.array(vector1).shape
    shape_v2 = numpy.array(vector2).shape
    assert shape_v1 == shape_v2, \
        f'Both 1D Vectors Dont have same shape, Actual Shape V1 - {shape_v1}, ' \
        f'Actual Shape V2 - {shape_v2}, Expected Shape For Both Vectors - {shape}'
    assert vector_dot == expected_value, \
        f'Dot Operator Fails, Actual - {vector_dot} does not match expected - {expected_value}'


@get_test_vectors_2D()
def test_dot_numpy_2D(vector1, shape1: tuple, vector2, shape2: tuple, dot_exp_shape: tuple, expected_value: int):
    print(f'vector1 - \n {vector1}')
    print(f'vector2 - \n {vector2}')
    vector_dot = dot.numpy_dot(vector1, vector2)
    print(f'vector dot - \n{vector_dot}')
    assert type(vector_dot) == numpy.ndarray
    assert isinstance(vector_dot, numpy.ndarray)
    shape_v1 = numpy.array(vector1).shape
    shape_v2 = numpy.array(vector2).shape
    assert shape_v1 == shape1, f'Actual Vector1 shape - {shape_v1} does not match expected shape - {shape1}'
    assert shape_v2 == shape2, f'Actual Vector1 shape - {shape_v2} does not match expected shape - {shape2}'
    vector_dot_shape = vector_dot.shape
    assert vector_dot_shape == dot_exp_shape, f'Vector Shape - {vector_dot_shape} does not match expected shape - {dot_exp_shape}'
    assert numpy.array_equal(
        vector_dot,
        expected_value), f'Dot Operator Fails, Actual - {vector_dot} does not match expected - {expected_value}'


@get_test_vectors_1D()
def test_dot_tensor_1D(vector1, vector2, shape: tuple, expected_value: int):
    print(f'vector1 - \n {vector1}')
    print(f'vector2 - \n {vector2}')
    vector_dot = dot.tensor_dot(vector1, vector2)
    print(f'vector dot - \n{vector_dot}')
    assert type(vector_dot) == torch.Tensor
    assert isinstance(vector_dot, torch.Tensor)
    shape_v1 = torch.tensor(vector1).shape
    shape_v2 = torch.tensor(vector2).shape
    assert shape_v1 == shape_v2, \
        f'Both 1D Vectors Dont have same shape, Actual Shape V1 - {shape_v1}, ' \
        f'Actual Shape V2 - {shape_v2}, Expected Shape For Both Vectors - {shape}'
    assert vector_dot == expected_value, \
        f'Dot Operator Fails, Actual - {vector_dot} does not match expected - {expected_value}'


@get_test_vectors_2D()
def test_dot_tensor_2D(vector1, shape1: tuple, vector2, shape2: tuple, dot_exp_shape: tuple, expected_value: int):
    print(f'vector1 - \n {vector1}')
    print(f'vector2 - \n {vector2}')
    vector_dot = dot.tensor_dot(vector1, vector2)
    print(f'vector dot - \n{vector_dot}')
    assert type(vector_dot) == torch.Tensor
    assert isinstance(vector_dot, torch.Tensor)
    shape_v1 = torch.tensor(vector1).shape
    shape_v2 = torch.tensor(vector2).shape
    assert shape_v1 == shape1, f'Actual Vector1 shape - {shape_v1} does not match expected shape - {shape1}'
    assert shape_v2 == shape2, f'Actual Vector1 shape - {shape_v2} does not match expected shape - {shape2}'
    vector_dot_shape = vector_dot.shape
    assert vector_dot_shape == dot_exp_shape, f'Vector Shape - {vector_dot_shape} does not match expected shape - {dot_exp_shape}'
    assert vector_dot.equal(
        torch.tensor(
            expected_value)), f'Dot Operator Fails, Actual - {vector_dot} does not match expected - {expected_value}'
