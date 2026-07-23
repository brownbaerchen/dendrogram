import pytest
from tempfile import TemporaryDirectory

from astrodendro.dendrogram import Dendrogram

from dendro.distributed_dendrogram_v3 import DistributedDendrogramV3
from dendro.utils import compare_dendrograms


@pytest.mark.parametrize("ntasks", [1, 2, 4])
@pytest.mark.parametrize("res", [32, 64])
@pytest.mark.parametrize("min_npix", [0])
@pytest.mark.parametrize("min_delta", [0, 0.1, 0.5])
@pytest.mark.parametrize("min_value", ["min"])
def test_1D_v3_pseudo_parallel(ntasks, res, min_npix, min_delta, min_value):
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

    # import matplotlib.pyplot as plt
    # from dendro.utils import plot_astrodendro_leaves

    # fig, axs = plt.subplots(2, max([ntasks, 2]))
    # local_dendrograms = (
    #     DistributedDendrogramV3.compute_local_dendrogram_pseudo_parallel(
    #         ntasks=ntasks, **kwargs
    #     )
    # )
    # for i, d in enumerate(local_dendrograms):
    #     plot_astrodendro_leaves(axs[0, i], x.numpy(), data.numpy(), d.trunk)
    # plot_astrodendro_leaves(axs[1, 0], x.numpy(), data.numpy(), dendrogram.trunk)
    # plot_astrodendro_leaves(
    #     axs[1, 1], x.numpy(), data.numpy(), reference_dendrogram.trunk
    # )
    # plt.show()

    compare_dendrograms(reference_dendrogram, dendrogram)


@pytest.mark.parametrize("ntasks", [1, 2, 4])
@pytest.mark.parametrize("res", [32, 64])
@pytest.mark.parametrize("n_peaks", [1, 2, 3, 4])
def test_2D_v3_pseudo_parallel(ntasks, res, n_peaks):
    from dendro.utils import get_2d_data

    _, _, data = get_2d_data(res, n_peaks)

    dendrogram = DistributedDendrogramV3.compute_pseudo_parallel(data.numpy(), ntasks)
    reference_dendrogram = Dendrogram.compute(data.numpy())

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


def test_example_pseudo_parallel():
    from astropy.io.fits import getdata
    import astrodendro
    import numpy as np

    data, header = getdata(
        f"{astrodendro.__file__[:-24]}/docs/PerA_Extn2MASS_F_Gal.fits", header=True
    )
    data = np.array(data, dtype=float)

    kwargs = {
        "min_value": 2.0,
        "min_delta": 1.0,
    }

    d_ref = astrodendro.Dendrogram.compute(data, **kwargs)
    d = DistributedDendrogramV3.compute_pseudo_parallel(data, 2, **kwargs)
    compare_dendrograms(d_ref, d)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    test_1D_v3_pseudo_parallel(4, 32, 0, 0.1, "min")
    # test_example_pseudo_parallel()
    # test_1D_v3_pseudo_parallel(2, 128)
    # test_2D_v3_pseudo_parallel(2, 32, 2)
