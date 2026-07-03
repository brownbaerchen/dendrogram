import heat as ht
import numpy as np


def get_1d_data(n):
    x = ht.linspace(0, 1, n)

    peaks = [0.25, 0.4, 0.5, 0.65]
    heights = [0.2, 0.8, 0.6, 0.7]

    data = ht.zeros_like(x)
    for peak, height in zip(peaks, heights):
        data += height * ht.exp(-((x - peak) ** 2) / 0.003)

    return x, data


def get_2d_data(n, n_peaks=-1):
    x = ht.linspace(0, 1, n)
    y = ht.linspace(0, 1, n)
    X, Y = ht.meshgrid(x, y, indexing="ij")
    X.resplit_(0)
    Y.resplit_(0)

    peaks = [(0.47, 0.4), (0.65, 0.65), (0.3, 0.7), (0.25, 0.25)]
    heights = [0.65, 0.6, 0.7, 0.65, 0.2]

    n_peaks = len(peaks) if n_peaks < 0 else n_peaks

    data = ht.zeros_like(X)
    for i in range(n_peaks):
        data += heights[i] * ht.exp(
            -((X - peaks[i][0]) ** 2 + (Y - peaks[i][1]) ** 2) / 0.03
        )

    return X, Y, data


def compare_dendrograms(ref_dendrogram, other_dendrogram):
    from dendro.distributed_dendrogram import shares_row

    n_structures1 = len([me for me in ref_dendrogram.all_structures])
    n_structures2 = len([me for me in other_dendrogram.all_structures])
    assert n_structures1 == n_structures2, (
        f"Got {n_structures1} structures in reference dendrogram, but {n_structures2} in other one"
    )

    for structure in ref_dendrogram.all_structures:
        corresponds_to = [
            ref_struct
            for ref_struct in other_dendrogram.all_structures
            if np.any(
                shares_row(np.array(structure._indices), np.array(ref_struct._indices))
            )
        ]
        assert len(corresponds_to) == 1, (
            f"Structure in reference dendrogram corresponds to {len(corresponds_to)} structures in the merged one"
        )
        assert np.allclose(
            np.sort(np.array(structure._indices).flatten()),
            np.sort(np.array(corresponds_to[0]._indices).flatten()),
        ), "Indices dont match between merged and reference structure"
