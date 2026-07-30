import pytest
from tempfile import TemporaryDirectory

from astrodendro.dendrogram import Dendrogram

from dendro.distributed_dendrogram_v3 import DistributedDendrogramV3
from dendro.utils import compare_dendrograms


@pytest.mark.parametrize("ntasks", [1, 2, 4, 16])
@pytest.mark.parametrize("res", [32, 33, 64, 256])
@pytest.mark.parametrize("min_npix", [0, 2, 6, 23])
@pytest.mark.parametrize("min_delta", [0, 2e-2, 0.1, 0.5])
@pytest.mark.parametrize("min_value", ["min", 0.2])
def test_1D_v3_pseudo_parallel(ntasks, res, min_npix, min_delta, min_value, show=False):
    from dendro.utils import get_1d_data

    x, data = get_1d_data(res)

    kwargs = {
        "data": data.numpy(),
        "min_npix": min_npix,
        "min_value": min_value,
        "min_delta": min_delta,
    }

    dendrogram = DistributedDendrogramV3.compute_pseudo_parallel(
        **kwargs, ntasks=ntasks
    )
    reference_dendrogram = Dendrogram.compute(**kwargs)

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from dendro.utils import plot_astrodendro_leaves

    # fig, axs = plt.subplots(2, max([ntasks, 2]))
    n_top = max([ntasks, 2])

    fig = plt.figure(figsize=(3 * n_top, 6))
    gs = GridSpec(2, n_top, figure=fig)

    # Top row: one axis per column
    axs_top = [fig.add_subplot(gs[0, i]) for i in range(n_top)]

    # Bottom row: two axes spanning half the width each
    axs_bottom = [
        fig.add_subplot(gs[1, : n_top // 2]),
        fig.add_subplot(gs[1, n_top // 2 :]),
    ]

    local_dendrograms = (
        DistributedDendrogramV3().compute_local_dendrogram_pseudo_parallel(
            ntasks=ntasks, **kwargs
        )
    )
    for i, d in enumerate(local_dendrograms):
        plot_astrodendro_leaves(axs_top[i], x.numpy(), data.numpy(), d.trunk)
    plot_astrodendro_leaves(axs_bottom[0], x.numpy(), data.numpy(), dendrogram.trunk)
    plot_astrodendro_leaves(
        axs_bottom[1], x.numpy(), data.numpy(), reference_dendrogram.trunk
    )
    if show:
        plt.show()

    compare_dendrograms(reference_dendrogram, dendrogram)


@pytest.mark.mpi(ranks=[2])
@pytest.mark.parametrize("res", [33, 64])
@pytest.mark.parametrize("min_npix", [0, 6])
@pytest.mark.parametrize("min_delta", [0, 0.1])
@pytest.mark.parametrize("min_value", ["min", 0.2])
def test_1D_v3(mpi_ranks, res, min_npix, min_delta, min_value):
    from dendro.utils import get_1d_data

    x, data = get_1d_data(res)

    ntasks = data.comm.size

    kwargs = {
        "min_npix": min_npix,
        "min_value": min_value,
        "min_delta": min_delta,
    }

    dendrogram = DistributedDendrogramV3.compute(data=data, **kwargs)
    reference_dendrogram = Dendrogram.compute(data=data.numpy(), **kwargs)

    import matplotlib.pyplot as plt
    from dendro.utils import plot_astrodendro_leaves

    fig, axs = plt.subplots(2, max([ntasks, 2]))
    _d = DistributedDendrogramV3()
    _d.data = data
    local_dendrograms = _d.compute_local_dendrogram(**kwargs)
    for i, d in enumerate([local_dendrograms]):
        plot_astrodendro_leaves(axs[0, i], x.numpy(), data.numpy(), d.trunk)
    plot_astrodendro_leaves(axs[1, 0], x.numpy(), data.numpy(), dendrogram.trunk)
    plot_astrodendro_leaves(
        axs[1, 1], x.numpy(), data.numpy(), reference_dendrogram.trunk
    )
    # plt.show()

    compare_dendrograms(reference_dendrogram, dendrogram)


@pytest.mark.parametrize("ntasks", [1, 2, 4])
@pytest.mark.parametrize("res", [32, 64])
@pytest.mark.parametrize("n_peaks", [1, 2, 3, 4])
@pytest.mark.parametrize("min_npix", [0, 16])
@pytest.mark.parametrize("min_delta", [0, 0.1])
@pytest.mark.parametrize("min_value", ["min", 0.2])
def test_2D_v3_pseudo_parallel(ntasks, res, n_peaks, min_npix, min_value, min_delta):
    from dendro.utils import get_2d_data

    _, _, data = get_2d_data(res, n_peaks)

    kwargs = {
        "min_npix": min_npix,
        "min_value": min_value,
        "min_delta": min_delta,
    }

    dendrogram = DistributedDendrogramV3.compute_pseudo_parallel(
        data.numpy(), ntasks, **kwargs
    )
    reference_dendrogram = Dendrogram.compute(data.numpy(), **kwargs)

    # import matplotlib.pyplot as plt
    # from dendro.utils import plot_astrodendro_tree_2D
    # fig, axs = plt.subplots(2, ntasks)
    # local_dendrograms = DistributedDendrogramV3.compute_local_dendrogram_pseudo_parallel(data.numpy(), ntasks)
    # for i, d in enumerate(local_dendrograms):
    #     plot_astrodendro_tree_2D(axs[0, i], d, d.trunk)
    # plot_astrodendro_tree_2D(axs[1, 0], dendrogram, dendrogram.trunk)
    # plot_astrodendro_tree_2D(axs[1, 1], reference_dendrogram, reference_dendrogram.trunk)
    # plt.show()

    compare_dendrograms(reference_dendrogram, dendrogram)


@pytest.mark.mpi(ranks=[1, 2])
@pytest.mark.parametrize("res", [32])
@pytest.mark.parametrize("n_peaks", [2, 3])
def test_2D_v3(mpi_ranks, res, n_peaks):
    from dendro.utils import get_2d_data

    _, _, data = get_2d_data(res, n_peaks)

    dendrogram = DistributedDendrogramV3.compute(data)
    reference_dendrogram = Dendrogram.compute(data.numpy())
    compare_dendrograms(reference_dendrogram, dendrogram)


@pytest.mark.mpi(ranks=[1, 2])
def test_2D_save_and_load(mpi_ranks):
    from dendro.utils import get_2d_data
    from astrodendro import Dendrogram

    _, _, data = get_2d_data(32, 4)

    dendrogram = DistributedDendrogramV3.compute(data)
    with TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/dendrogram.fits"
        dendrogram.save_to(output_path)
        compare_to = Dendrogram.load_from(output_path)
        compare_dendrograms(compare_to, dendrogram)


@pytest.mark.parametrize("ntasks", [1, 2])  # , 2, 4])
@pytest.mark.parametrize("min_value", [2])
@pytest.mark.parametrize("min_delta", [0])  # , 0.02])
@pytest.mark.parametrize("min_npix", [0])  # , 4])
def test_example_pseudo_parallel(ntasks, min_value, min_delta, min_npix, plot=False):
    from astropy.io.fits import getdata
    import astrodendro
    import numpy as np

    data, header = getdata(
        f"{astrodendro.__file__[:-24]}/docs/PerA_Extn2MASS_F_Gal.fits", header=True
    )
    data = np.array(data, dtype=float)

    kwargs = {
        "min_value": min_value,
        "min_delta": min_delta,
        "min_npix": min_npix,
    }

    d_ref = astrodendro.Dendrogram.compute(data, **kwargs)
    d = DistributedDendrogramV3.compute_pseudo_parallel(data, ntasks, **kwargs)

    def ting(d_ref, d):
        for idx_ref in np.unique(d_ref.index_map):
            if idx_ref < 0:
                continue

            idx = d.index_map[*[me[0] for me in np.nonzero(d_ref.index_map == idx_ref)]]

            struct_ref = d_ref._structures_dict[idx_ref]
            struct = d._structures_dict[idx]

            indices = np.sort(np.array(struct._indices).flatten())
            indices_ref = np.sort(np.array(struct_ref._indices).flatten())

            structs_equal = len(indices) == len(indices_ref)
            if structs_equal:
                structs_equal = np.allclose(indices, indices_ref)
            if not structs_equal:
                import matplotlib.pyplot as plt

                v = np.zeros_like(d.index_map)
                print(np.unique(v, return_counts=True))
                v[*(np.array(struct._indices)).T] += 1
                print(np.unique(v, return_counts=True))
                v[*(np.array(struct_ref._indices)).T] -= -1
                print(np.unique(v, return_counts=True))
                breakpoint()

                fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 3))
                axs[0].contour(v, levels=[0.5, 1.5], colors=["black", "orange"])
                d_ref.plotter().plot_contour(axs[1], structure=struct_ref)
                d.plotter().plot_contour(axs[2], structure=struct)
                plt.show()

    ting(d_ref, d)

    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        d_ref.plotter().plot_contour(axs[0])
        d.plotter().plot_contour(axs[1])
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        d_ref.plotter().plot_tree(axs[0])
        d.plotter().plot_tree(axs[1])
        plt.show()
    compare_dendrograms(d_ref, d)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # test_1D_v3_pseudo_parallel(2, 32, 6, 0.0, 0.0)
    # test_1D_v3_pseudo_parallel(16, 32, 0, 0.0, 0.2, show=True)
    test_example_pseudo_parallel(
        ntasks=2, min_value=2, min_delta=0, min_npix=0, plot=True
    )
