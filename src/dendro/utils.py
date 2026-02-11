import heat as ht


def get_1d_data(n):
    x = ht.linspace(0, 1, n)

    peaks = [0.25, 0.4, 0.5, 0.65]
    heights = [0.2, 0.8, 0.6, 0.7]

    data = ht.zeros_like(x)
    for peak, height in zip(peaks, heights):
        data += height * ht.exp(-((x - peak) ** 2) / 0.003)

    return x, data


def get_2d_data(n):
    x = ht.linspace(0, 1, n)
    y = ht.linspace(0, 1, n)
    X, Y = ht.meshgrid(x, y, indexing="ij")

    peaks = [(0.25, 0.25), (0.47, 0.4), (0.65, 0.65), (0.3, 0.7)]
    heights = [0.2, 0.65, 0.6, 0.7, 0.65]

    data = ht.zeros_like(X)
    for peak, height in zip(peaks, heights):
        data += height * ht.exp(-((X - peak[0]) ** 2 + (Y - peak[1]) ** 2) / 0.03)

    return X, Y, data
