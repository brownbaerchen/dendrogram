import pytest

from astrodendro.dendrogram import Dendrogram

from dendro.distributed_dendrogram_v3 import DistributedDendrogramV3
from dendro.utils import compare_dendrograms


@pytest.mark.parametrize("ntasks", [1, 2, 4])
@pytest.mark.parametrize("res", [32, 64])
def test_1D_v3_pseudo_parallel(ntasks, res):
    from dendro.utils import get_1d_data

    x, data = get_1d_data(res)

    dendrogram = DistributedDendrogramV3.compute_pseudo_parallel(data.numpy(), ntasks)
    reference_dendrogram = Dendrogram.compute(data.numpy())

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2, ntasks)
    # local_dendrograms = DistributedDendrogramV3.compute_local_dendrogram_pseudo_parallel(data.numpy(), ntasks)
    # for i, d in enumerate(local_dendrograms):
    #     plot_astrodendro_leaves(axs[0, i], x.numpy(), data.numpy(), d.trunk)
    # plot_astrodendro_leaves(axs[1, 0], x.numpy(), data.numpy(), dendrogram.trunk)
    # plot_astrodendro_leaves(axs[1, 1], x.numpy(), data.numpy(), reference_dendrogram.trunk)
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


if __name__ == "__main__":
    test_1D_v3_pseudo_parallel(2, 64)
