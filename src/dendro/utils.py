import heat as ht
import numpy as np


def get_1d_data(n):
    x = ht.linspace(0, 1, n, split=0)

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
            f"Structure {structure.idx} in reference dendrogram corresponds to {len(corresponds_to)} structures {[me.idx for me in corresponds_to]} in the merged one"
        )
        assert len(np.unique(corresponds_to[0]._indices, axis=0)) == len(
            corresponds_to[0]._indices
        ), f"Structure {corresponds_to[0].idx} has non-unique indices"
        assert len(structure._indices) == len(corresponds_to[0]._indices), (
            f"Structure {corresponds_to[0].idx} has different length from reference structure {structure.idx}"
        )
        assert np.allclose(
            np.sort(np.array(structure._indices).flatten()),
            np.sort(np.array(corresponds_to[0]._indices).flatten()),
        ), r"Indices don\'t match between merged and reference structure"


def plot_astrodendro_leaves(ax, x, data, leaves, level=0, plot_children=True):
    markers = {0: ".", 1: "x", 2: ">", 3: "o", 4: "<"}

    if level == 0:
        ax.plot(x, data, color="black")

    for leaf in leaves:
        ax.scatter(
            np.array(x)[leaf._indices],
            np.array(data)[leaf._indices],
            marker=markers.get(level, "."),
        )
        if plot_children:
            plot_astrodendro_leaves(
                ax=ax, x=x, data=data, leaves=leaf._children, level=level + 1
            )


def plot_astrodendro_tree_2D(
    ax,
    dendrogram,
    leaves,
    _plotter=None,
    plot_children=True,
    min_level=0,
    max_level=np.inf,
):
    if _plotter is None:
        _plotter = dendrogram.plotter()
        data = (
            dendrogram.data
            if isinstance(dendrogram.data, np.ndarray)
            else dendrogram.data.numpy()
        )
        ax.imshow(data, cmap="Reds")

    for leaf in leaves:
        if leaf.level >= min_level:
            _plotter.plot_contour(ax, structure=leaf)
        if plot_children and leaf.level <= max_level:
            plot_astrodendro_tree_2D(
                ax,
                dendrogram,
                leaf.children,
                _plotter=_plotter,
                min_level=min_level,
                max_level=max_level,
            )


def plot(ax, dendrogram, leaves, plot_children=True):
    data = dendrogram.data
    if isinstance(data, ht.DNDarray):
        data = data.numpy()

    if data.ndim == 1:
        x = np.arange(data.shape[0])
        plot_astrodendro_leaves(ax, x, data, leaves, plot_children=plot_children)
    elif data.ndim == 2:
        remove_trunk = False
        if not hasattr(dendrogram, "_trunk"):
            dendrogram._trunk = leaves
            remove_trunk = True

        def _convert_to_list(s):
            s._indices = list(s._indices)
            s._values = list(s._values)
            s._peak = None
            s._descendants = None
            s._dendrogram = dendrogram
            s._tree_index = None
            s._level = 0
            for _s in s.children:
                _convert_to_list(_s)

        for s in leaves:
            _convert_to_list(s)

        plot_astrodendro_tree_2D(ax, dendrogram, leaves, plot_children=plot_children)
        if remove_trunk:
            delattr(dendrogram, "_trunk")

        def _convert_to_numpy(s):
            s._indices = np.array(s._indices)
            s._values = np.array(s._values)
            for _s in s.children:
                _convert_to_numpy(_s)

        for s in leaves:
            _convert_to_numpy(s)
    else:
        raise NotImplementedError
