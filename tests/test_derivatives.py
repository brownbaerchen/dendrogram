import pytest


def test_derivative_2D():
    import numpy as np
    from dendro.derivative import compute_derivative

    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 2)

    X, Y = np.meshgrid(x, y, indexing="ij")

    data = X
    expect_derivative = np.ones_like(data)
    derivative = compute_derivative([X, Y], data, axis=0)
    assert np.allclose(expect_derivative, derivative)

    data = Y
    expect_derivative = np.ones_like(data)
    derivative = compute_derivative([X, Y], data, axis=1)
    assert np.allclose(expect_derivative, derivative)

    data = X + Y
    expect_derivative = np.ones_like(data)
    derivative = compute_derivative([X, Y], data, axis=1)
    assert np.allclose(expect_derivative, derivative)

    data = X * Y
    expect_derivative = X.copy()
    derivative = compute_derivative([X, Y], data, axis=1)
    assert np.allclose(expect_derivative, derivative)

    data = X * Y
    expect_derivative = Y.copy()
    derivative = compute_derivative([X, Y], data, axis=0)
    assert np.allclose(expect_derivative, derivative)

    data = X**2
    expect_derivative = 2 * X
    derivative = compute_derivative([X, Y], data, axis=0)
    assert np.allclose(expect_derivative[:, 1:-1], derivative[:, 1:-1])

    data = Y**2 * X**2
    expect_derivative = 2 * Y * X**2
    derivative = compute_derivative([X, Y], data, axis=1)
    assert np.allclose(expect_derivative[1:-1, :], derivative[1:-1, :])


@pytest.mark.parametrize("f", [1, 2, 4])
def test_find_extrema(f):
    import numpy as np
    from dendro.derivative import find_extrema, compute_derivative

    x = np.linspace(0, 2 * np.pi, 256)
    data = np.sin(f * x) ** 2
    deriv = compute_derivative([x], data)

    extrema = find_extrema(deriv)
    expect_extrema = ((f * x) % (np.pi / 2)) <= 1.8 * x[1]
    assert np.allclose(expect_extrema, expect_extrema)


if __name__ == "__main__":
    test_derivative_2D()
