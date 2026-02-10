import numpy as np


def get_slice(ndim):
    return [slice(None, None, None) for _ in range(ndim)]


def compute_derivative(grids, data, axis=0):
    s1, s2, s3 = get_slice(data.ndim), get_slice(data.ndim), get_slice(data.ndim)
    s1[axis] = slice(1, data.shape[axis] - 1, None)
    s2[axis] = slice(None, data.shape[axis] - 2, None)
    s3[axis] = slice(2, None, None)

    derivative = np.empty_like(data)
    derivative[*s1] = (data[*s2] - data[*s3]) / (grids[axis][*s2] - grids[axis][*s3])

    # compute derivative at left boundary
    s1, s2 = get_slice(data.ndim), get_slice(data.ndim)
    s1[axis] = slice(0, 1, None)
    s2[axis] = slice(1, 0, -1)
    derivative[*s1] = (data[*s1] - data[*s2]) / (grids[axis][*s1] - grids[axis][*s2])

    # compute derivative at right boundary
    s1, s2 = get_slice(data.ndim), get_slice(data.ndim)
    s1[axis] = slice(data.shape[axis] - 1, data.shape[axis], None)
    s2[axis] = slice(data.shape[axis] - 2, data.shape[axis] - 1, None)
    derivative[*s1] = (data[*s1] - data[*s2]) / (grids[axis][*s1] - grids[axis][*s2])

    return derivative


def find_extrema(derivative, axis=0):
    s1, s2, s3 = (
        get_slice(derivative.ndim),
        get_slice(derivative.ndim),
        get_slice(derivative.ndim),
    )
    s1[axis] = slice(1, derivative.shape[axis] - 1, None)
    s2[axis] = slice(None, derivative.shape[axis] - 2, None)
    s3[axis] = slice(1, derivative.shape[axis] - 1, None)

    sign = np.sign(derivative)

    extrema = np.ones_like(derivative).astype(bool)
    extrema[*s1] = sign[*s2] * sign[*s3] == -1
    return extrema
