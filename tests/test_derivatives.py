import pytest

def test_derivative():
    import numpy as np
    from dendro.derivative import compute_derivative

    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 2)

    X, Y = np.meshgrid(x, y, indexing='ij')

    data = X
    expect_derivative = np.ones_like(data)
    derivative = compute_derivative([X, Y], data, axis=0)
    assert np.allclose(expect_derivative, derivative)
    
    data = Y
    expect_derivative = np.ones_like(data)
    derivative = compute_derivative([X, Y], data, axis=1)
    assert np.allclose(expect_derivative, derivative)

    data = X+Y
    expect_derivative = np.ones_like(data)
    derivative = compute_derivative([X, Y], data, axis=1)
    assert np.allclose(expect_derivative, derivative)

    data = X*Y
    expect_derivative = X.copy()
    derivative = compute_derivative([X, Y], data, axis=1)
    assert np.allclose(expect_derivative, derivative)

    data = X*Y
    expect_derivative = Y.copy()
    derivative = compute_derivative([X, Y], data, axis=0)
    assert np.allclose(expect_derivative, derivative)

    data = X**2
    expect_derivative = 2 * X
    derivative = compute_derivative([X, Y], data, axis=0)
    assert np.allclose(expect_derivative[:, 1:-1], derivative[:, 1:-1])

    data = Y**2 * X**2
    expect_derivative = 2 * X
    derivative = compute_derivative([X, Y], data, axis=0)
    assert np.allclose(expect_derivative[:, 1:-1], derivative[:, 1:-1])


if __name__ == '__main__':
    test_derivative()
