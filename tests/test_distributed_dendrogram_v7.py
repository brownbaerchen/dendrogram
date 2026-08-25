import pytest

from dendro.distributed_dendrogram_v7 import DistributedDendrogramV7, Dendrogram, ht


@pytest.mark.parametrize("ntasks", [1, 2, 4])
@pytest.mark.parametrize("res", [32, 33, 64])
@pytest.mark.parametrize("min_npix", [0])
@pytest.mark.parametrize("min_delta", [0])
@pytest.mark.parametrize("min_value", ["min"])
def test_1D_pseudo_parallel(ntasks, res, min_npix, min_delta, min_value, show=False):
    from dendro.utils import get_1d_data, compare_dendrograms

    x, data = get_1d_data(res)

    kwargs = {
        "data": data.numpy(),
        "min_npix": min_npix,
        "min_value": min_value,
        "min_delta": min_delta,
    }

    dendrogram = DistributedDendrogramV7.compute_pseudo_parallel(
        data=data, ntasks=ntasks
    )
    reference_dendrogram = Dendrogram.compute(**kwargs)

    if show:
        import matplotlib.pyplot as plt

        from dendro.utils import plot_astrodendro_leaves

        fig, axs = plt.subplots(1, 2)
        plot_astrodendro_leaves(axs[0], x.numpy(), data.numpy(), dendrogram.trunk)
        plot_astrodendro_leaves(
            axs[1], x.numpy(), data.numpy(), reference_dendrogram.trunk
        )
        plt.show()

    compare_dendrograms(reference_dendrogram, dendrogram)


@pytest.mark.parametrize("ntasks", [1, 2, 4])
@pytest.mark.parametrize("res", [32, 64])
@pytest.mark.parametrize("n_peaks", [1, 2, 3, 4])
@pytest.mark.parametrize("min_npix", [0])
@pytest.mark.parametrize("min_delta", [0])
@pytest.mark.parametrize("min_value", ["min"])
def test_2D_v3_pseudo_parallel(
    ntasks, res, n_peaks, min_npix, min_value, min_delta, show=False
):
    from dendro.utils import get_2d_data, compare_dendrograms

    _, _, data = get_2d_data(res, n_peaks)

    kwargs = {
        "min_npix": min_npix,
        "min_value": min_value,
        "min_delta": min_delta,
    }

    reference_dendrogram = Dendrogram.compute(data.numpy(), **kwargs)
    dendrogram = DistributedDendrogramV7.compute_pseudo_parallel(
        data.numpy(),
        ntasks,  # **kwargs
    )

    if show:
        import matplotlib.pyplot as plt
        from dendro.utils import plot_astrodendro_tree_2D

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        plot_astrodendro_tree_2D(axs[0], dendrogram, dendrogram.trunk)
        plot_astrodendro_tree_2D(
            axs[1], reference_dendrogram, reference_dendrogram.trunk
        )
        plt.show()

    compare_dendrograms(reference_dendrogram, dendrogram)


@pytest.mark.skip
@pytest.mark.parametrize("ntasks", [1, 2, 4])
@pytest.mark.parametrize("min_value", [-1e9])
@pytest.mark.parametrize("min_delta", [0])
@pytest.mark.parametrize("min_npix", [0])  # 10
def test_example_pseudo_parallel(ntasks, min_value, min_delta, min_npix):
    from astropy.io.fits import getdata
    import astrodendro
    import numpy as np
    from dendro.utils import compare_dendrograms

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
    d = DistributedDendrogramV7.compute_pseudo_parallel(data, ntasks)  # , **kwargs)
    compare_dendrograms(d_ref, d)


if __name__ == "__main__":
    import logging

    if ht.comm.rank == 0:
        logger = logging.getLogger("Dendrogram").setLevel(logging.DEBUG)
        # logger..basicConfig(level=logging.DEBUG)
    # test_1D_pseudo_parallel(32, 64, 0, 0, 0, show=True)
    test_2D_v3_pseudo_parallel(4, 64, 4, 0, 0, 0, show=True)
    # test_example_pseudo_parallel(2, -1e9, 0, 0)
